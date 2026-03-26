# API para exponer el script de consultas

## 1) Instalar dependencias

```bash
pip install -r requirements_api.txt
```

## 2) Variables de entorno opcionales

```bash
export MARKDOWN_DIR=./out_md
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_EMBED_MODEL=embeddinggemma
export OLLAMA_CHAT_MODEL=qwen3:8b
export HAYSTACK_ENABLED=true
export USE_OLLAMA_GENERATION=false
```

## 3) Levantar la API

```bash
uvicorn qa_api_app:app --host 0.0.0.0 --port 8000 --reload
```

## 4) Endpoints

### Health
```bash
curl http://localhost:8000/health
```

### Reconstruir índice
```bash
curl -X POST http://localhost:8000/build
```

### Consultar
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What commands do I need to install external-dns?",
    "top_k": 10
  }'
```

## 5) Ejemplo desde una interfaz web

```html
<script>
async function consultar() {
  const resp = await fetch('http://localhost:8000/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: 'What commands do I need to install external-dns?',
      top_k: 10
    })
  });

  const data = await resp.json();
  console.log(data);
  document.getElementById('respuesta').textContent = data.answer;
}
</script>

<button onclick="consultar()">Consultar</button>
<pre id="respuesta"></pre>
```
