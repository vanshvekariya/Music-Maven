import { useState } from 'react';
import { Search, Sparkles, Loader2 } from 'lucide-react';
import { Button } from './ui/Button';
import { Input } from './ui/Input';
import { cn } from '@/utils/cn';

const LANG_OPTIONS = [
  { value: '', label: 'Any language' },
  { value: 'en', label: 'English (en)' },
  { value: 'pt', label: 'Portuguese (pt)' },
  { value: 'es', label: 'Spanish (es)' },
  { value: 'fr', label: 'French (fr)' },
  { value: 'de', label: 'German (de)' },
  { value: 'it', label: 'Italian (it)' },
  { value: 'ko', label: 'Korean (ko)' },
  { value: 'ja', label: 'Japanese (ja)' },
];

export function QueryInput({
  onSubmit,
  isLoading,
  examples,
  sessionMemoryActive = false,
  showExamples = true,
}) {
  const [query, setQuery] = useState('');
  const [langFilter, setLangFilter] = useState('');
  const [maxResults, setMaxResults] = useState(10);
  const [useKG, setUseKG] = useState(true);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      const q = query.trim();
      setQuery('');
      onSubmit(q, {
        langFilter: langFilter || undefined,
        maxResults: Math.min(100, Math.max(1, Number(maxResults) || 10)),
        useKG,
      });
    }
  };

  const handleExampleClick = (exampleQuery) => {
    setQuery('');
    onSubmit(exampleQuery, {
      langFilter: langFilter || undefined,
      maxResults: Math.min(100, Math.max(1, Number(maxResults) || 10)),
      useKG,
    });
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

      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
        <div className="flex flex-wrap items-center gap-2">
          <label htmlFor="max-results" className="text-muted-foreground whitespace-nowrap">
            Max results
          </label>
          <input
            id="max-results"
            type="number"
            min={1}
            max={100}
            value={maxResults}
            onChange={(e) => setMaxResults(e.target.value)}
            disabled={isLoading}
            className="w-20 rounded-md border border-input bg-background px-2 py-2 text-foreground shadow-sm focus:outline-none focus:ring-2 focus:ring-primary"
          />
          <span className="text-xs text-muted-foreground">(vector path)</span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <label htmlFor="lang-filter" className="text-muted-foreground whitespace-nowrap">
            Vector language
          </label>
          <select
            id="lang-filter"
            value={langFilter}
            onChange={(e) => setLangFilter(e.target.value)}
            disabled={isLoading}
            className="rounded-md border border-input bg-background px-3 py-2 text-foreground shadow-sm focus:outline-none focus:ring-2 focus:ring-primary"
          >
            {LANG_OPTIONS.map((o) => (
              <option key={o.value || 'any'} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </div>
        <label htmlFor="use-kg" className="flex items-center gap-2 text-muted-foreground whitespace-nowrap">
          <input
            id="use-kg"
            type="checkbox"
            checked={useKG}
            onChange={(e) => setUseKG(e.target.checked)}
            disabled={isLoading}
            className="h-4 w-4 rounded border-input text-primary focus:ring-primary"
          />
          Use KG (instant)
        </label>
        <span className="text-xs text-muted-foreground w-full sm:w-auto">
          Language filter applies to vector search only; KG toggle controls KG routing.
        </span>
        {sessionMemoryActive && (
          <span className="text-xs text-muted-foreground w-full border-t border-border/60 pt-2 mt-1">
            Session memory is on: the server keeps a short{' '}
            <strong className="text-foreground font-medium">rolling summary</strong> of what you
            asked and what was returned (not full transcripts), so follow-ups like &quot;show me
            more&quot; can avoid repeating the same list. <strong>New chat</strong> clears the on-screen
            thread and server memory for this tab.
          </span>
        )}
      </div>

      {/* Example Queries — hide after chat starts (ChatGPT-style) */}
      {showExamples && examples && examples.length > 0 && (
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground font-medium">
            Try these examples:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
            {examples.slice(0, 9).map((example, index) => {
              const categoryColors = {
                'KG (instant)': 'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200',
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
