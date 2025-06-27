// Importa dotenv para leer las variables de entorno (.env)
import * as dotenv from "dotenv";
dotenv.config();

// Importa el conector de LangChain para Groq
import { ChatGroq } from "@langchain/groq";

async function main() {
  // Crea una instancia del modelo Groq
  const llm = new ChatGroq({
    apiKey: process.env.GROQ_API_KEY,
    // Modelo llama que vamos a usar
    model: "llama3-70b-8192", 
  });

  // Pregunta de prueba
  const input = "¿Quién es el presidente de España?";

  console.log("Enviando pregunta:", input);

  // Envia el mensaje al LLM y espera la respuesta
  const response = await llm.invoke(input);

  // Muestra la respuesta en consola
  console.log("Respuesta del modelo:", response.content);
}

// Ejecuta la función main y muestra cualquier error
main().catch(console.error);
