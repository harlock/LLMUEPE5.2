import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, BookOpen, AlertCircle } from 'lucide-react';

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: '¡Hola! Soy el Copiloto de Soporte Técnico para **UEPE 5.2**. ¿En qué te puedo ayudar hoy con la instalación, configuración o administración de la plataforma?',
      sources: []
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Auto-scroll al final de los mensajes
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Función para limpiar la respuesta cruda de Python/Haystack
  const cleanHaystackResponse = (rawContent) => {
    let text = rawContent;
    if (typeof text !== 'string') return '';

    // Detecta si es un objeto ChatMessage que expone Python
    if (text.includes('ChatMessage(') && text.includes('text=')) {
      // Extrae solo el texto útil usando Regex
      const match = text.match(/text=(['"])([\s\S]*?)\1\)\]/);
      if (match && match[2]) {
        text = match[2]
          .replace(/\\n/g, '\n') // Convierte los \n literales en saltos de línea
          .replace(/\\'/g, "'")  // Limpia comillas simples escapadas
          .replace(/\\"/g, '"')  // Limpia comillas dobles escapadas
          .replace(/\\\\/g, '\\'); // Limpia barras invertidas
      }
    }
    return text;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userQuery = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userQuery }]);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: userQuery,
          temperature: 0.3 // Temperatura baja para respuestas técnicas más precisas
        }),
      });

      if (!response.ok) {
        throw new Error(`Error del servidor: ${response.statusText}`);
      }

      const data = await response.json();

      // Limpiamos la respuesta de la IA antes de guardarla en el estado
      const cleanedAnswer = cleanHaystackResponse(data.answer);

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: cleanedAnswer,
        sources: data.sources || []
      }]);

    } catch (error) {
      console.error("Error al consultar la API:", error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        isError: true,
        content: 'Hubo un error al intentar conectar con el servidor de UEPE. Verifica que la API (app.py) esté corriendo en el puerto 8000.'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Función para renderizar Markdown básico (código y negritas) útil para las respuestas técnicas
  const renderMessageContent = (content) => {
    // Separar bloques de código (```)
    const parts = content.split('```');

    return parts.map((part, index) => {
      // Si el índice es impar, es un bloque de código
      if (index % 2 === 1) {
        const lines = part.trim().split('\n');
        const code = lines.length > 1 && !lines[0].includes(' ') ? lines.slice(1).join('\n') : part.trim();
        return (
          <div key={index} className="my-3 bg-slate-900 rounded-md overflow-hidden shadow-sm border border-slate-700">
            <div className="flex items-center px-4 py-1 bg-slate-800 border-b border-slate-700 text-xs text-slate-400 font-mono">
              Code Snippet
            </div>
            <pre className="p-4 overflow-x-auto text-sm text-slate-50 font-mono">
              <code>{code}</code>
            </pre>
          </div>
        );
      }

      // Procesar texto normal: saltos de línea, negritas (**texto**) y código en línea (`código`)
      return (
        <div key={index} className="whitespace-pre-wrap text-slate-700 leading-relaxed">
          {part.split('\n').map((line, i) => {
            const boldParts = line.split(/(\*\*.*?\*\*)/g);
            return (
              <span key={i}>
                {boldParts.map((bp, j) => {
                  if (bp.startsWith('**') && bp.endsWith('**')) {
                    return <strong key={j} className="font-semibold text-slate-900">{bp.slice(2, -2)}</strong>;
                  }

                  // Nuevo: Detectar y estilizar código en línea (comillas simples invertidas)
                  const inlineCodeParts = bp.split(/(`[^`]+`)/g);
                  return inlineCodeParts.map((icp, k) => {
                    if (icp.startsWith('`') && icp.endsWith('`')) {
                      return (
                        <code key={`${j}-${k}`} className="bg-slate-200 border border-slate-300 text-indigo-700 px-1.5 py-0.5 rounded-md font-mono text-[0.85em] font-medium shadow-sm">
                          {icp.slice(1, -1)}
                        </code>
                      );
                    }
                    return icp;
                  });
                })}
                {i !== part.split('\n').length - 1 && <br />}
              </span>
            );
          })}
        </div>
      );
    });
  };

  return (
    <div className="flex flex-col h-screen bg-slate-50 font-sans">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 shadow-sm px-6 py-4 flex items-center justify-between sticky top-0 z-10">
        <div className="flex items-center gap-3">
          <div className="bg-indigo-600 p-2 rounded-lg shadow-sm">
            <Bot className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-800 leading-tight">UEPE 5.2 Copilot</h1>
            <p className="text-xs text-slate-500 font-medium tracking-wide uppercase">Technical Support Assistant</p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-sm text-slate-500 bg-slate-100 px-3 py-1.5 rounded-full font-medium">
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
          System Online
        </div>
      </header>

      {/* Chat Area */}
      <main className="flex-1 overflow-y-auto p-4 sm:p-6 w-full max-w-4xl mx-auto">
        <div className="space-y-6">
          {messages.map((msg, idx) => (
            <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>

              {/* Avatar Assistant */}
              {msg.role === 'assistant' && (
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center border border-indigo-200 mt-1">
                  <Bot className="w-5 h-5 text-indigo-600" />
                </div>
              )}

              {/* Message Bubble */}
              <div className={`max-w-[85%] rounded-2xl px-5 py-4 shadow-sm ${msg.role === 'user'
                  ? 'bg-indigo-600 text-white rounded-tr-sm'
                  : msg.isError
                    ? 'bg-red-50 border border-red-200 rounded-tl-sm'
                    : 'bg-white border border-slate-200 rounded-tl-sm'
                }`}>

                {/* Error State */}
                {msg.isError ? (
                  <div className="flex items-start gap-2 text-red-600">
                    <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                    <p className="text-sm font-medium">{msg.content}</p>
                  </div>
                ) : (
                  <>
                    {/* Content */}
                    <div className={msg.role === 'user' ? 'text-white' : ''}>
                      {msg.role === 'user' ? msg.content : renderMessageContent(msg.content)}
                    </div>

                    {/* Sources (Only for assistant and if they exist) */}
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="mt-4 pt-4 border-t border-slate-100">
                        <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                          <BookOpen className="w-4 h-4" />
                          <span>Fuentes consultadas</span>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {msg.sources.map((source, sIdx) => (
                            <span key={sIdx} className="inline-flex items-center px-2.5 py-1 rounded-md bg-slate-100 text-slate-600 text-xs font-medium border border-slate-200">
                              {source}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>

              {/* Avatar User */}
              {msg.role === 'user' && (
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center border border-slate-300 mt-1">
                  <User className="w-5 h-5 text-slate-600" />
                </div>
              )}
            </div>
          ))}

          {/* Loading State */}
          {isLoading && (
            <div className="flex gap-4 justify-start">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center border border-indigo-200 mt-1">
                <Bot className="w-5 h-5 text-indigo-600" />
              </div>
              <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-sm px-5 py-4 shadow-sm flex items-center gap-3">
                <Loader2 className="w-5 h-5 text-indigo-500 animate-spin" />
                <span className="text-sm text-slate-500 font-medium">Analizando documentación de UEPE...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input Area */}
      <footer className="bg-white border-t border-slate-200 p-4 w-full">
        <div className="max-w-4xl mx-auto">
          <form onSubmit={handleSubmit} className="relative flex items-center">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ej. ¿Cuáles son los pasos para instalar UEPE en AWS?"
              disabled={isLoading}
              className="w-full pl-5 pr-14 py-4 bg-slate-100 border-transparent focus:bg-white focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 rounded-xl text-slate-700 placeholder-slate-400 transition-all duration-200 shadow-inner disabled:opacity-60 disabled:cursor-not-allowed"
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="absolute right-2 p-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 text-white rounded-lg transition-colors duration-200 shadow-sm"
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
          <div className="text-center mt-3">
            <p className="text-xs text-slate-400">
              El asistente está estrictamente limitado a la documentación técnica. Las respuestas generadas son informativas.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}