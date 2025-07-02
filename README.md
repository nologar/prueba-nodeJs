# prueba-nodeJs

Este es un proyecto en Node.js donde se implementa un agente mediante langchain y langraph que decide si buscar la información que necesita mediante la tool tavily o responder directamente.
El grafo permite decidir si el LLM responde directamente o usa una herramienta como Tavily para hacer búsquedas en la web. 

## Cómo usarlo

1. Crea un archivo `.env` en la raíz del proyecto con las claves necesarias:

GROQ_API_KEY=tu_api_key_de_groq
TAVILY_API_KEY=tu_api_key_de_tavily


2. Instala las dependencias:

npm install


3. Ejecuta el bot en tu terminal:

node main.js 


4. Abre tu navegador en [http://localhost:3000](http://localhost:3000)

Y empieza a chatear con JSBot sobre lo que quieras.


## Esquema del Grafo

![Diagrama del grafo](./Grafo.png)
