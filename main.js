// Cargamos las variables de entorno de env
import * as dotenv from "dotenv";
dotenv.config();

// Importamos las librerías necesarias
import { ChatGroq } from "@langchain/groq";
import { StateGraph, START, END } from "@langchain/langgraph";
import { z } from "zod";
import { TavilySearch } from "@langchain/tavily";
import readline from "readline";

// Flag de debug
const debug = true; // Con esto se controla si queremos mostrar los logs o solo el resultado final
// Número máximo de mensajes de contexto
const MAX_HISTORY_MESSAGES = 10;

// Declaramos la memoria de la conversación
let chatHistory = [];

// -----------------------------------------------
// Función mejorada para extraer JSON
// Intenta parsear todo el string primero.
// Si falla, busca el primer bloque JSON válido, incluso anidado.
// Lanza errores específicos para mayor control.
// -----------------------------------------------
function extractJSON(str) {
  try {
    return JSON.parse(str);
  } catch (e) {
    // Busca el primer bloque JSON anidado
    const jsonRegex = /{(?:[^{}]|{(?:[^{}]|{[^{}]*})*})*}/s;
    const match = str.match(jsonRegex);
    if (match) {
      try {
        return JSON.parse(match[0]);
      } catch (e2) {
        throw new Error("JSON inválido dentro del bloque");
      }
    }
    throw new Error("No se encontró JSON en la respuesta del LLM.");
  }
}

// Función para leer input del usuario desde terminal
function promptUser(question) {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      rl.close();
      resolve(answer);
    });
  });
}

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
  // Prompt reforzado para indicar al LLM que jamás escriba texto fuera del JSON.
  const systemMessage = `
Eres un asistente inteligente llamado JSBot y ordenado. Tu objetivo es ayudar al usuario de forma precisa y segura. Responde siempre en el idioma en que te preguntan.

**IMPORTANTE CRÍTICO**:
- ¡NUNCA escribas texto FUERA del bloque JSON!
- Si incluyes texto adicional, el sistema fallará.
- Ejemplo INCORRECTO: "Pienso que... {\"action\": ...}"
- Ejemplo CORRECTO: {\"action\": ...}

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
`;

  // Definimos el nodo del LLM
  const chatbotNode = async (state) => {
    // Si estamos en modo debug, mostramos el estado actual
    if (debug) console.log("\n🤖 Ejecutando LLM...");

    // Construimos el historial para mantener el contexto
    let historyText = "";
    if (chatHistory.length > 0) {
      // Nos quedamos con las últimas N entradas para no superar el límite de tokens
      const trimmedHistory = chatHistory.slice(-MAX_HISTORY_MESSAGES);
      historyText = "CONVERSACIÓN ANTERIOR:\n";
      for (const message of trimmedHistory) {
        historyText += `- ${message.role === "user" ? "Usuario" : "Asistente"}: ${message.content}\n`;
      }
    }

    // Creamos el prompt incluyendo el historial y la pregunta actual
    let prompt = `${systemMessage}\n\n${historyText}\n\nPREGUNTA ACTUAL:\nUsuario: ${state.input}`;

    if (state.intermediateSteps?.length) {
      prompt += `\n\nInformación obtenida de herramientas:\n${JSON.stringify(state.intermediateSteps, null, 2)}`;
    }

    // Instanciamos un modelo de Groq LLM
    const llm = new ChatGroq({
      apiKey: process.env.GROQ_API_KEY,
      model: "deepseek-r1-distill-llama-70b", // Modelo de Groq deepseek-r1-distill-llama-70b.
      temperature: 0.3, // Controla la aleatoriedad de las respuestas
      maxTokens: 1024, // Máximo de tokens en la respuesta del LLM
      maxRetries: 2, // Reintentos automáticos en caso de errores transitorios
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
      // Usamos la función extractJSON 
      parsed = extractJSON(response.content);

      // Validación adicional de estructura
      // Solo se permiten las acciones "finish" o "use_tool"
      if (!parsed.action || !(parsed.action === "finish" || parsed.action === "use_tool")) {
        throw new Error("Estructura JSON inválida o acción desconocida.");
      }
    } catch (e) {
      console.error("❌ Error parseando JSON:", e);

      //  En caso de error, limpiamos el historial para evitar loops de contexto corrupto
      chatHistory = [];

      return {
        ...state,
        result: "Disculpa, tuve un error interno. ¿Podrías reformular tu pregunta?",
        next: "END",
      };
    }

    // Mostramos si se ha decidido usar una herramienta o responder directamente
    if (parsed.action === "use_tool") {
      if (debug) console.log("🔧 El LLM ha decidido usar la tool:", parsed.tool);
      return {
        //retornamos el estado actualizado con la llamada a la herramienta
        ...state,
        tool_call: {
          tool_name: parsed.tool,
          args: parsed.tool_input,
        },
        next: "tools",
      };
    } else if (parsed.action === "finish") {
      if (debug) console.log("✅ El LLM ha decidido responder directamente.");

      // Guardamos en el historial
      chatHistory.push({ role: "user", content: state.input });
      chatHistory.push({ role: "assistant", content: parsed.answer });

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

  // Mensaje de bienvenida
  console.log(`
👋 ¡Hola! Soy JSBot, tu asistente inteligente.

Puedo:
- responder preguntas generales
- buscar información en Internet si no sé algo (uso Tavily Search)
- mantener el contexto de nuestra conversación

Para finalizar la conversación en cualquier momento, escribe: salir
`);
  // Entramos en bucle para seguir conversando
  while (true) {
    const userInput = await promptUser("\n> ");

    if (userInput.toLowerCase() === "salir") {
      console.log("¡Hasta luego!");
      break;
    }

    const finalState = await executor.invoke({
      input: userInput,
    });

    if (debug) {
      console.log("\n Estado final:", finalState);
    }

    console.log("\n 🤖 Respuesta :", finalState.result);
  }
}

// Llama a main y captura errores
main().catch(console.error);
