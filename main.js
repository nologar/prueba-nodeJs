import * as dotenv from "dotenv";
dotenv.config();

import { ChatGroq } from "@langchain/groq";
import { StateGraph, START, END } from "@langchain/langgraph";
import { z } from "zod";

async function main() {
  // Definimos el esquema de entrada y salida del grafo
  const schema = z.object({
    input: z.string(),           // lo que le preguntas al bot
    result: z.string().optional(), // la respuesta final
  });

  // Definimos el nodo LLM
  const llmNode = async (state) => {
    console.log("Nodo LLM recibió:", state.input);

    // Instanciamos un modelo Groq LLM
    const llm = new ChatGroq({
      apiKey: process.env.GROQ_API_KEY,
      model: "llama3-70b-8192", 
    });

    // Llamamos al modelo para hacer una pregunta
    const response = await llm.invoke(state.input);

    console.log("Respuesta desde el nodo LLM:", response.content);

    // Devuelve un nuevo state, mergeando el resultado
    return {
      ...state,
      result: response.content,
    };
  };

  // Creamos el grafo pasándole el schema
  const graph = new StateGraph(schema);

  // Añadimos el LLM como un nodo en el grafo
  graph.addNode("llm", llmNode);

  // Definimos el punto de entrada y la salida
  graph.addEdge(START, "llm");
  graph.addEdge("llm", END);

  // Compilamos el grafo
  const executor = graph.compile();

  // Ejecutamos el grafo con una pregunta
  const finalState = await executor.invoke({
    input: "¿Quién es el presidente de España?",
  });
  // Mostramos el estado final del grafo y la respuesta para hacer debug
  console.log("Estado final del grafo:", finalState);
  console.log("Respuesta final del grafo:", finalState.result);
}

// Llama a main y captura errores
main().catch(console.error);
