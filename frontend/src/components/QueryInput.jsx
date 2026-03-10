import { useState } from 'react';
import { Search, Sparkles, Loader2 } from 'lucide-react';
import { Button } from './ui/Button';
import { Input } from './ui/Input';
import { cn } from '@/utils/cn';

export function QueryInput({ onSubmit, isLoading, examples }) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSubmit(query.trim());
    }
  };

  const handleExampleClick = (exampleQuery) => {
    setQuery(exampleQuery);
    onSubmit(exampleQuery);
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6">
      {/* Main Search Bar */}
      <form onSubmit={handleSubmit} className="relative">
        <div className="relative flex items-center">
          <Search className="absolute left-4 h-5 w-5 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Ask anything about music... (e.g., 'Top artists by popularity' or 'High energy songs in English')"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={isLoading}
            className="pl-12 pr-32 h-14 text-base shadow-lg border-2 focus:border-primary"
          />
          <Button
            type="submit"
            disabled={!query.trim() || isLoading}
            className="absolute right-2 h-10"
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Sparkles className="mr-2 h-4 w-4" />
                Search
              </>
            )}
          </Button>
        </div>
      </form>

      {/* Example Queries */}
      {examples && examples.length > 0 && (
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground font-medium">
            Try these examples:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
            {examples.slice(0, 9).map((example, index) => {
              const categoryColors = {
                'SQL':    'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300',
                'Vector': 'bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300',
                'Hybrid': 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300',
              };
              const badgeClass = categoryColors[example.category] || 'bg-muted text-muted-foreground';
              return (
              <button
                key={index}
                onClick={() => handleExampleClick(example.query)}
                disabled={isLoading}
                className={cn(
                  'text-left p-3 rounded-lg border border-border bg-card hover:bg-accent hover:border-primary transition-all duration-200 group',
                  'disabled:opacity-50 disabled:cursor-not-allowed'
                )}
              >
                <div className="flex items-start gap-2">
                  <Sparkles className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-foreground group-hover:text-primary transition-colors line-clamp-2">
                      {example.query}
                    </p>
                    <span className={cn('inline-block text-xs font-semibold px-2 py-0.5 rounded-full mt-1', badgeClass)}>
                      {example.category}
                    </span>
                  </div>
                </div>
              </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
