import { useState, useEffect, useRef } from 'react';
import { Toaster, toast } from 'react-hot-toast';
import {
  Music,
  Sparkles,
  Database,
  Activity,
  AlertCircle,
  Info,
} from 'lucide-react';
import { QueryInput } from './components/QueryInput';
import { ResultsDisplay } from './components/ResultsDisplay';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/Card';
import { Badge } from './components/ui/Badge';
import { Button } from './components/ui/Button';
import { apiService } from './services/api';

function newTurnId() {
  return typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : `turn-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function App() {
  /** Each turn = one user question + assistant reply (ChatGPT-style thread). */
  const [turns, setTurns] = useState([]);
  const [examples, setExamples] = useState([]);
  const [systemInfo, setSystemInfo] = useState(null);
  const [healthStatus, setHealthStatus] = useState(null);
  const [showInfo, setShowInfo] = useState(false);
  const [sessionId, setSessionId] = useState(() =>
    typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID()
      : `sess-${Date.now()}`
  );
  const bottomRef = useRef(null);

  const isLoading = turns.some((t) => t.loading);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [turns]);

  // Load examples and system info on mount
  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      // Load examples
      const examplesData = await apiService.getExamples();
      setExamples(examplesData);

      // Check health
      const health = await apiService.checkHealth();
      setHealthStatus(health);

      // Load system info
      const info = await apiService.getSystemInfo();
      setSystemInfo(info);
    } catch (error) {
      console.error('Failed to load initial data:', error);
      toast.error('Failed to connect to backend. Please ensure the API is running.');
    }
  };

  const handleQuery = async (query, options = {}) => {
    const turnId = newTurnId();
    setTurns((prev) => [
      ...prev,
      { id: turnId, query, loading: true, response: null, error: null },
    ]);

    try {
      const result = await apiService.processQuery(
        query,
        options.maxResults ?? 10,
        options.langFilter || null,
        options.useKG ?? true,
        sessionId,
        {
          memoryTurns: options.memoryTurns ?? 5,
          memoryMaxChars: options.memoryMaxChars ?? 2000,
        }
      );
      setTurns((prev) =>
        prev.map((t) =>
          t.id === turnId
            ? {
                ...t,
                loading: false,
                response: result,
                error: result.success ? null : result.error || 'Query failed',
              }
            : t
        )
      );

      if (result.success) {
        toast.success('Query processed successfully!');
      } else {
        toast.error(result.error || 'Query processing failed');
      }
    } catch (error) {
      console.error('Query error:', error);
      const msg =
        error.response?.data?.detail ||
        'Failed to process query. Please try again.';
      setTurns((prev) =>
        prev.map((t) =>
          t.id === turnId ? { ...t, loading: false, response: null, error: msg } : t
        )
      );
      toast.error(msg);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      <Toaster position="top-right" />

      {/* Header */}
      <header className="bg-white border-b shadow-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <Music className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-foreground">
                  Music Maven
                </h1>
                <p className="text-sm text-muted-foreground">
                  Music Information Retrieval
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {healthStatus && (
                <Badge
                  variant={
                    healthStatus.status === 'healthy' ? 'success' : 'destructive'
                  }
                >
                  <Activity className="h-3 w-3 mr-1" />
                  {healthStatus.status}
                </Badge>
              )}
              <Button
                variant="outline"
                size="sm"
                onClick={async () => {
                  try {
                    await apiService.clearConversation(sessionId);
                  } catch {
                    /* non-fatal */
                  }
                  setTurns([]);
                  setSessionId(
                    typeof crypto !== 'undefined' && crypto.randomUUID
                      ? crypto.randomUUID()
                      : `sess-${Date.now()}`
                  );
                  toast.success('New chat — history cleared');
                }}
                title="Clear this conversation on screen and server memory"
              >
                New chat
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowInfo(!showInfo)}
              >
                <Info className="h-4 w-4 mr-2" />
                System Info
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* System Info Panel */}
      {showInfo && systemInfo && (
        <div className="border-b bg-muted/30 backdrop-blur-sm">
          <div className="container mx-auto px-4 py-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">System Information</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <p className="text-sm font-medium mb-2">Available Agents:</p>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(systemInfo.agents || {}).map(
                      ([key, agent]) => (
                        <Badge key={key} variant="outline">
                          {agent.type === 'sql' && (
                            <Database className="h-3 w-3 mr-1" />
                          )}
                          {agent.type === 'vector' && (
                            <Sparkles className="h-3 w-3 mr-1" />
                          )}
                          {agent.name}
                        </Badge>
                      )
                    )}
                  </div>
                </div>
                <div>
                  <p className="text-sm font-medium mb-2">Configuration:</p>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                    {Object.entries(systemInfo.configuration || {}).map(
                      ([key, value]) => (
                        <div key={key} className="p-2 bg-muted rounded">
                          <p className="text-muted-foreground capitalize">
                            {key.replace(/_/g, ' ')}
                          </p>
                          <p
                            className={`font-medium ${
                              value !== null && typeof value === 'object'
                                ? 'text-xs break-words'
                                : 'truncate'
                            }`}
                          >
                            {value !== null && typeof value === 'object'
                              ? JSON.stringify(value)
                              : String(value ?? '')}
                          </p>
                        </div>
                      )
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {/* Main: scrollable thread + sticky input (ChatGPT-style) */}
      <main className="flex flex-col min-h-[calc(100vh-8rem)]">
        <div className="flex-1 container mx-auto px-4 py-6 space-y-6 max-w-6xl w-full">
          {turns.length === 0 && (
            <div className="text-center space-y-4 py-8">
              <h2 className="text-4xl md:text-5xl font-bold tracking-tight">
                Explore Music4All
              </h2>
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                Ask questions in natural language and get intelligent insights from
                109,000 songs using multi-agent AI
              </p>
              <p className="text-sm text-muted-foreground">
                Use the bar below — each reply stays in the thread until you start a new chat.
              </p>
            </div>
          )}

          {turns.map((turn) => (
            <section
              key={turn.id}
              className="space-y-4 scroll-mt-24 border-b border-border/40 pb-8 last:border-0"
            >
              <div className="flex justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-tr-md bg-primary text-primary-foreground px-4 py-3 shadow-sm">
                  <p className="text-sm font-medium whitespace-pre-wrap break-words">
                    {turn.query}
                  </p>
                </div>
              </div>

              {turn.loading && (
                <Card className="border-2 border-primary/20 max-w-6xl mx-auto">
                  <CardContent className="py-10">
                    <div className="flex flex-col items-center justify-center gap-4">
                      <div className="relative">
                        <div className="h-14 w-14 rounded-full border-4 border-primary/20 border-t-primary animate-spin" />
                        <Sparkles className="h-5 w-5 text-primary absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
                      </div>
                      <p className="text-sm text-muted-foreground">
                        AI agents are analyzing the data…
                      </p>
                    </div>
                  </CardContent>
                </Card>
              )}

              {!turn.loading && turn.error && (
                <Card className="border-destructive/40 bg-destructive/5 max-w-6xl mx-auto">
                  <CardContent className="py-4 flex items-start gap-3">
                    <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                    <p className="text-sm text-destructive">{turn.error}</p>
                  </CardContent>
                </Card>
              )}

              {!turn.loading && turn.response && (
                <ResultsDisplay response={turn.response} />
              )}
            </section>
          ))}

          <div ref={bottomRef} aria-hidden="true" className="h-1" />
        </div>

        <div className="sticky bottom-0 z-40 border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80 shadow-[0_-4px_24px_-8px_rgba(0,0,0,0.08)]">
          <div className="container mx-auto px-4 py-4 max-w-6xl">
            <QueryInput
              onSubmit={handleQuery}
              isLoading={isLoading}
              examples={examples}
              sessionMemoryActive
              showExamples={turns.length === 0}
            />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t mt-16">
        <div className="container mx-auto px-4 py-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-muted-foreground">
            <p>Music Maven · Music4All Dataset · 109k Songs</p>
            <div className="flex items-center gap-4">
              <Badge variant="outline">
                <Database className="h-3 w-3 mr-1" />
                SQL Agent
              </Badge>
              <Badge variant="outline">
                <Sparkles className="h-3 w-3 mr-1" />
                Vector Agent
              </Badge>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
