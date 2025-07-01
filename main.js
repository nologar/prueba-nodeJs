// Cargamos las variables de entorno de env
import * as dotenv from "dotenv";
dotenv.config();

// Importamos las librerías necesarias
import { ChatGroq } from "@langchain/groq";
import { StateGraph, START, END } from "@langchain/langgraph";
import { z } from "zod";
import { TavilySearch } from "@langchain/tavily";

// Flag de debug
const debug = false; // Con esto se controla si queremos mostrar los logs o solo el resultado final

async function main() {
  // Definimos el esquema de entrada y salida del grafo
  const schema = z.object({
    input: z.string(), // lo que le preguntas al bot
    intermediateSteps: z.array(z.any()).optional(),
    result: z.string().optional(), // respuesta final del bot
    // Si el LLM decide usar una herramienta, aquí se define la llamada
    tool_call: z.object({
      tool_name: z.string(),
      args: z.any(),
    }).optional(),
    next: z.string().optional(),
  });

  // Configuramos la herramienta Tavily para búsqueda web
  const tavilyTool = new TavilySearch({
    apiKey: process.env.TAVILY_API_KEY,
  });

  // Prompt que guía al LLM 
  const systemMessage = `
Eres un asistente inteligente y ordenado. Tu objetivo es ayudar al usuario de forma precisa y segura.

- Siempre debes responder en formato JSON válido.
- No escribas texto fuera del bloque JSON (ni explicaciones, ni comentarios, ni etiquetas como <think>).
- Si conoces la respuesta con seguridad, devuélvela directamente así:

{"action":"finish","answer":"...respuesta..."}

- Si no estás completamente seguro de la respuesta o crees que necesitas información actualizada, debes usar la herramienta Tavily. Para ello, responde así:

{
  "action":"use_tool",
  "tool":"tavily_search",
  "tool_input":{"query":"...texto a buscar..."}
}

- Si usas Tavily, incluye en tu respuesta final el enlace de la fuente principal que hayas usado (URL) para respaldar tu información.

Ejemplo de respuesta directa:

{"action":"finish","answer":"El presidente de España es Pedro Sánchez desde 2018."}

Ejemplo de petición para usar Tavily:

{
  "action":"use_tool",
  "tool":"tavily_search",
  "tool_input":{"query":"Último resultado del Real Madrid"}
}
`;

  // Definimos el nodo del LLM
  const chatbotNode = async (state) => {
    // Si estamos en modo debug, mostramos el estado actual
    if (debug) console.log("\n🤖 Ejecutando LLM...");

    let prompt = `${systemMessage}\n\nPregunta del usuario: ${state.input}`;

    if (state.intermediateSteps?.length) {
      prompt += `\n\nInformación obtenida de herramientas:\n${JSON.stringify(state.intermediateSteps, null, 2)}`;
    }

    // Instanciamos un modelo de Groq LLM
    const llm = new ChatGroq({
      apiKey: process.env.GROQ_API_KEY,
      model: "qwen-qwq-32b", // Modelo de Groq qwen.
      temperature: 0.3, // Controla la aleatoriedad de las respuestas
      maxTokens: 1024, // Máximo de tokens en la respuesta del LLM
    });

    let response;
    try {
      // Invocamos el LLM con el prompt generado
      response = await llm.invoke(prompt);
    } catch (error) {
      console.error("❌ Error al invocar el LLM:", error);
      return {
        ...state,
        result: "Error: No se pudo invocar el LLM.",
        next: "END",
      };
    }
    // Si estamos en modo debug, mostramos la respuesta cruda del LLM
    if (debug) console.log("➡️ Respuesta RAW del LLM:\n", response.content);

    let parsed;
    try {
      // Busca bloque JSON multi-línea en toda la respuesta
      const regex = /{[\s\S]*}/m;
      const match = response.content.match(regex);

      if (!match) {
        throw new Error("No se encontró JSON en la respuesta del LLM.");
      }

      parsed = JSON.parse(match[0]);

      if (!parsed.action) {
        throw new Error("El JSON devuelto no contiene el campo 'action'.");
      }
    } catch (e) {
      console.error("❌ Error parseando JSON:", e);
      return {
        ...state,
        result: "Error: El LLM no devolvió JSON válido.",
        next: "END",
      };
    }

    // Mostramos si se ha decidido usar una herramienta o responder directamente
    if (parsed.action === "use_tool") {
      if (debug) console.log("🔧 El LLM ha decidido usar la tool:", parsed.tool);
      return {
        //returnamos el estado actualizado con la llamada a la herramienta
        ...state,
        tool_call: {
          tool_name: parsed.tool,
          args: parsed.tool_input,
        },
        next: "tools",
      };
    } else if (parsed.action === "finish") {
      if (debug) console.log("✅ El LLM ha decidido responder directamente.");
      return {
        ...state,
        result: parsed.answer,
        next: "END",
      };
    } else {
      console.error("⚠️ Acción desconocida en la respuesta del LLM:", parsed);
      return {
        ...state,
        result: "Error: respuesta desconocida del LLM.",
        next: "END",
      };
    }
  };

  // Nodo de la herramienta
  const toolsNode = async (state) => {
    if (debug) console.log("\n🛠 Ejecutando tools...");
    // Si no hay tool_call, retornamos al chatbot
    let toolResult = null;

    if (state.tool_call?.tool_name === "tavily_search") {
      try {
        toolResult = await tavilyTool.invoke({
          query: state.tool_call.args.query,
        });
        if (debug) console.log("🔎 Resultado Tavily:", toolResult);
      } catch (error) {
        console.error("❌ Error ejecutando Tavily:", error);
        toolResult = "Error: No se pudo ejecutar Tavily.";
      }
    } else {
      toolResult = "Error: Tool no reconocida.";
    }

    const steps = state.intermediateSteps || [];
    steps.push({
      tool_used: state.tool_call.tool_name,
      tool_output: toolResult,
    });

    return {
      ...state,
      intermediateSteps: steps,
      next: "chatbot",
    };
  };

  // Creamos el grafo pasándole el schema.
  const graph = new StateGraph(schema);
  graph.addNode("chatbot", chatbotNode);
  graph.addNode("tools", toolsNode);
  graph.addEdge(START, "chatbot");
  graph.addConditionalEdges("chatbot", (state) => state.next);
  graph.addEdge("tools", "chatbot");
  graph.addEdge("chatbot", END);
  // Compilamos el grafo para que esté listo para ejecutar.
  const executor = graph.compile();

  // Ejecutamos el grafo con una pregunta inicial.
  const finalState = await executor.invoke({
    input: "¿Dime el tiempo para hoy en Valencia?",
  });

  if (debug) {
    console.log("\n✅ Estado final:", finalState);
  }

  console.log("\n✅ Respuesta final del grafo:", finalState.result);
}

// Llama a main y captura errores
main().catch(console.error);
