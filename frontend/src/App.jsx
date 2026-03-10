import { useState, useEffect } from 'react';
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

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [examples, setExamples] = useState([]);
  const [systemInfo, setSystemInfo] = useState(null);
  const [healthStatus, setHealthStatus] = useState(null);
  const [showInfo, setShowInfo] = useState(false);

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

  const handleQuery = async (query) => {
    setIsLoading(true);
    setResponse(null);

    try {
      const result = await apiService.processQuery(query);
      setResponse(result);
      
      if (result.success) {
        toast.success('Query processed successfully!');
      } else {
        toast.error(result.error || 'Query processing failed');
      }
    } catch (error) {
      console.error('Query error:', error);
      toast.error(
        error.response?.data?.detail || 
        'Failed to process query. Please try again.'
      );
    } finally {
      setIsLoading(false);
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
                          <p className="font-medium truncate">{value}</p>
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

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 space-y-8">
        {/* Hero Section */}
        {!response && (
          <div className="text-center space-y-4 py-12">

            <h2 className="text-4xl md:text-5xl font-bold tracking-tight">
              Explore Music4All
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Ask questions in natural language and get intelligent insights from
              109,000 songs using multi-agent AI
            </p>
          </div>
        )}

        {/* Query Input */}
        <QueryInput
          onSubmit={handleQuery}
          isLoading={isLoading}
          examples={examples}
        />

        {/* Loading State */}
        {isLoading && (
          <Card className="border-2 border-primary/20">
            <CardContent className="py-12">
              <div className="flex flex-col items-center justify-center gap-4">
                <div className="relative">
                  <div className="h-16 w-16 rounded-full border-4 border-primary/20 border-t-primary animate-spin" />
                  <Sparkles className="h-6 w-6 text-primary absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
                </div>
                <div className="text-center space-y-2">
                  <p className="text-lg font-medium">Processing your query...</p>
                  <p className="text-sm text-muted-foreground">
                    AI agents are analyzing the data
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Results */}
        {!isLoading && response && <ResultsDisplay response={response} />}

        {/* Empty State */}
        {!isLoading && !response && (
          <div className="text-center py-12 space-y-4">
            <div className="inline-flex p-4 bg-muted rounded-full">
              <AlertCircle className="h-8 w-8 text-muted-foreground" />
            </div>
            <div className="space-y-2">
              <h3 className="text-lg font-semibold">Ready to explore</h3>
              <p className="text-muted-foreground">
                Ask about artists, genres, audio features, or song details
              </p>
            </div>
          </div>
        )}
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
