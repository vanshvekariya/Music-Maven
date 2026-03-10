import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Music,
  Mic2,
  Disc3,
  TrendingUp,
  Tag,
  Sparkles,
  Database,
  Zap,
  Clock,
  Globe,
  Activity,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/Card';
import { Badge } from './ui/Badge';

export function ResultsDisplay({ response }) {
  if (!response) return null;

  const { answer, metadata, results, processing_time } = response;

  const getAgentIcon = (agentType) => {
    switch (agentType) {
      case 'sql':
        return <Database className="h-4 w-4" />;
      case 'vector':
        return <Sparkles className="h-4 w-4" />;
      case 'hybrid':
        return <Zap className="h-4 w-4" />;
      default:
        return <Sparkles className="h-4 w-4" />;
    }
  };

  const getAgentColor = (agentType) => {
    switch (agentType) {
      case 'sql':
        return 'default';
      case 'vector':
        return 'secondary';
      case 'hybrid':
        return 'success';
      default:
        return 'outline';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="w-full max-w-6xl mx-auto space-y-6"
    >
      {/* Metadata Bar */}
      {metadata && (
        <div className="flex flex-wrap items-center gap-3 p-4 bg-muted/50 rounded-lg border">
          {metadata.query_type && (
            <Badge variant={getAgentColor(metadata.query_type)}>
              {getAgentIcon(metadata.query_type)}
              <span className="ml-1.5 capitalize">{metadata.query_type}</span>
            </Badge>
          )}
          {metadata.agents_used && metadata.agents_used.length > 0 && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Agents:</span>
              {metadata.agents_used.map((agent, idx) => (
                <Badge key={idx} variant="outline" className="text-xs">
                  {agent}
                </Badge>
              ))}
            </div>
          )}
          {metadata.confidence !== undefined && (
            <Badge variant="outline">
              <TrendingUp className="h-3 w-3 mr-1" />
              {(metadata.confidence * 100).toFixed(0)}% Confidence
            </Badge>
          )}
          {processing_time && (
            <Badge variant="outline">
              <Clock className="h-3 w-3 mr-1" />
              {processing_time.toFixed(2)}s
            </Badge>
          )}
        </div>
      )}

      {/* Answer Section */}
      <Card className="shadow-sm">
        <CardHeader className="border-b">
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            <span>Answer</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="prose prose-sm max-w-none dark:prose-invert">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {answer}
            </ReactMarkdown>
          </div>
        </CardContent>
      </Card>

      {/* Song Results Grid */}
      {results && results.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-xl font-semibold flex items-center gap-2">
            <Music className="h-5 w-5 text-primary" />
            Songs ({results.length})
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {results.map((result, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <SongCard result={result} />
              </motion.div>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
}

function SongCard({ result }) {
  const pct = (val) =>
    val !== undefined && val !== null
      ? `${Math.round(val * 100)}%`
      : null;

  const tags = Array.isArray(result.tags)
    ? result.tags
    : typeof result.tags === 'string'
    ? result.tags.split(',').map((t) => t.trim()).filter(Boolean)
    : [];

  const genres = Array.isArray(result.genres)
    ? result.genres
    : typeof result.genres === 'string'
    ? result.genres.split(',').map((g) => g.trim()).filter(Boolean)
    : [];

  return (
    <Card className="hover:shadow-lg transition-shadow duration-200 h-full">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            <CardTitle className="text-base line-clamp-1">
              {result.song_name || 'Unknown Song'}
            </CardTitle>
            {result.artist && (
              <p className="text-sm text-muted-foreground flex items-center gap-1 mt-1">
                <Mic2 className="h-3 w-3 flex-shrink-0" />
                {result.artist}
              </p>
            )}
            {result.album && (
              <p className="text-xs text-muted-foreground flex items-center gap-1 mt-0.5">
                <Disc3 className="h-3 w-3 flex-shrink-0" />
                {result.album}
              </p>
            )}
          </div>
          {/* Similarity score from vector search */}
          {result.score !== undefined && (
            <Badge variant="secondary" className="flex-shrink-0">
              {(result.score * 100).toFixed(0)}%
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-3">
        {/* Audio feature pills */}
        <div className="grid grid-cols-2 gap-2">
          {result.popularity !== undefined && result.popularity !== null && (
            <div className="flex items-center gap-1.5 text-sm">
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
              <span className="font-medium">{result.popularity}</span>
              <span className="text-muted-foreground text-xs">popularity</span>
            </div>
          )}
          {result.tempo !== undefined && result.tempo !== null && (
            <div className="flex items-center gap-1.5 text-sm">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <span className="font-medium">{Math.round(result.tempo)}</span>
              <span className="text-muted-foreground text-xs">BPM</span>
            </div>
          )}
          {pct(result.energy) && (
            <div className="flex items-center gap-1.5 text-sm">
              <Activity className="h-4 w-4 text-muted-foreground" />
              <span className="font-medium">{pct(result.energy)}</span>
              <span className="text-muted-foreground text-xs">energy</span>
            </div>
          )}
          {pct(result.danceability) && (
            <div className="flex items-center gap-1.5 text-sm">
              <Sparkles className="h-4 w-4 text-muted-foreground" />
              <span className="font-medium">{pct(result.danceability)}</span>
              <span className="text-muted-foreground text-xs">dance</span>
            </div>
          )}
        </div>

        {/* Language + mode */}
        <div className="flex flex-wrap gap-2">
          {result.lang && (
            <Badge variant="outline" className="text-xs">
              <Globe className="h-3 w-3 mr-1" />
              {result.lang}
            </Badge>
          )}
          {result.mode !== undefined && result.mode !== null && (
            <Badge variant="outline" className="text-xs">
              {result.mode === 1 ? 'Major' : 'Minor'}
            </Badge>
          )}
          {result.has_lyrics && (
            <Badge variant="outline" className="text-xs">
              Lyrics
            </Badge>
          )}
        </div>

        {/* Genres */}
        {genres.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {genres.slice(0, 3).map((g, idx) => (
              <Badge key={idx} variant="secondary" className="text-xs">
                {g}
              </Badge>
            ))}
            {genres.length > 3 && (
              <Badge variant="secondary" className="text-xs">
                +{genres.length - 3}
              </Badge>
            )}
          </div>
        )}

        {/* Tags */}
        {tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            <Tag className="h-3 w-3 text-muted-foreground mt-0.5" />
            {tags.slice(0, 3).map((tag, idx) => (
              <span key={idx} className="text-xs text-muted-foreground">
                {tag}{idx < Math.min(tags.length, 3) - 1 ? ',' : ''}
              </span>
            ))}
            {tags.length > 3 && (
              <span className="text-xs text-muted-foreground">
                +{tags.length - 3} more
              </span>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
