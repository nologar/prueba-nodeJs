// Cargamos las variables de entorno de env
import * as dotenv from "dotenv";
dotenv.config();

// Importamos las librerías necesarias
import { ChatGroq } from "@langchain/groq";
import { StateGraph, START, END } from "@langchain/langgraph";
import { z } from "zod";
import { TavilySearch } from "@langchain/tavily";

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

  // Prompt que guaida al LLM
  const systemMessage = `
Eres un asistente inteligente.

- Si puedes contestar directamente, responde en JSON así:
{"action":"finish","answer":"...respuesta..."}

- Si necesitas buscar en internet, responde en JSON así:
{
  "action":"use_tool",
  "tool":"tavily_search",
  "tool_input":{"query":"...texto a buscar..."}
}

No escribas ningún texto fuera del bloque JSON.

Incluye en tu respuesta la URL de la fuente si usas Tavily.
`;

  // Definimos el nodo del LLM
  const chatbotNode = async (state) => {
    console.log("\n🤖 Ejecutando LLM...");

    let prompt = `${systemMessage}\n\nPregunta del usuario: ${state.input}`;

    if (state.intermediateSteps?.length) {
      prompt += `\n\nInformación obtenida de herramientas:\n${JSON.stringify(state.intermediateSteps, null, 2)}`;
    }
    //Instanciamos un modelo de groq LLM
    const llm = new ChatGroq({
      apiKey: process.env.GROQ_API_KEY,
      model: "qwen-qwq-32b", // Modelo de groq qwen.
      temperature: 0.3, // Controla la aleatoriedad de las respuestas
      maxTokens: 1024, // Máximo de tokens en la respuesta del LLM
    });
    
    // Llamamos al modelo para hacer una pregunta
    const response = await llm.invoke(prompt);

    console.log("➡️ Respuesta RAW del LLM:\n", response.content);

    let parsed;
    try {
      // Buscamos el JSON en la respuesta del LLM
      const regex = /{[\s\S]*}/;
      const match = response.content.match(regex);
      if (match) {
        parsed = JSON.parse(match[0]);
      } else {
        throw new Error("No se encontró JSON en la respuesta del LLM.");
      }
      // Validamos que la respuesta tenga la estructura esperada
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
      console.log("🔧 El LLM ha decidido usar la tool:", parsed.tool);
      return {
        ...state,
        tool_call: {
          tool_name: parsed.tool,
          args: parsed.tool_input,
        },
        next: "tools",
      };
    } else if (parsed.action === "finish") {
      console.log("✅ El LLM ha decidido responder directamente.");
      return {
        ...state,
        result: parsed.answer,
        next: "END",
      };
    } else {
      // Devuelve un nuevo state, mergeando el resultado
      return {
        ...state,
        result: "Error: respuesta desconocida del LLM.",
        next: "END",
      };
    }
  };

  // Nodo de la herramienta
  const toolsNode = async (state) => {
    //Si se llama a una herramienta se muestra por consola
    console.log("\n🛠 Ejecutando tools...");

    let toolResult = null;

    if (state.tool_call?.tool_name === "tavily_search") {
      toolResult = await tavilyTool.call({
        query: state.tool_call.args.query,
      });
      console.log("🔎 Resultado Tavily:", toolResult);
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
  //Añadimos el LLM como un nodo en el grafo
  graph.addNode("chatbot", chatbotNode);
  // Añadimos el nodo de herramientas
  graph.addNode("tools", toolsNode);
  // Definimos el punto de entrada y de salida del grafo.
  graph.addEdge(START, "chatbot");
  // Añadimos las transiciones entre nodos
  graph.addConditionalEdges("chatbot", (state) => state.next);
  graph.addEdge("tools", "chatbot");

  graph.addEdge("chatbot", END);
  // Compilamos el grafo.
  const executor = graph.compile();

  // Ejecutamos el grafo con una pregunta inicial.
  const finalState = await executor.invoke({
    input: "¿Dime el tiempo para hoy en Valencia?",
  });
  // Mostramos el estado final del grafo y la respuesta para hacer debug
  console.log("\n✅ Estado final:", finalState);
  console.log("\n✅ Respuesta final del grafo:", finalState.result);
}
// Llama a main y captura errores
main().catch(console.error);
