import { useState, useEffect } from 'react'
import { Target, Brain, Calendar, RefreshCw, Loader2, PlusCircle, Activity, ShieldAlert, Zap, Crosshair } from 'lucide-react'
import api from '../services/api'
import { useRefreshOnUpdate, useSeasonReset } from '../hooks/useRealTimeUpdates'

export default function Dashboard() {
  const [overview, setOverview] = useState<any>(null)
  const [rtData, setRtData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  
  const [scrapingResults, setScrapingResults] = useState(false)
  const [scrapingMatches, setScrapingMatches] = useState(false)
  const [creatingSeason, setCreatingSeason] = useState(false)
  const [scrapeMessage, setScrapeMessage] = useState<string | null>(null)

  const loadDashboard = async () => {
    try {
      const [data, rtResponse] = await Promise.all([
        api.getOverview().catch(() => null),
        api.get('/api/realtime/dashboard').catch(() => ({ data: null }))
      ])
      
      if (data) setOverview(data)
      if (rtResponse?.data) setRtData(rtResponse.data)
      
    } catch (error) {
      console.error('Failed to load dashboard:', error)
    } finally {
      setLoading(false)
    }
  }

  const resetDashboardData = () => {
    setOverview(null)
    setRtData(null)
    setLoading(true)
    setTimeout(() => loadDashboard(), 100)
  }

  useRefreshOnUpdate(loadDashboard)
  useSeasonReset(resetDashboardData)

  useEffect(() => {
    loadDashboard()
    const interval = setInterval(loadDashboard, 10000)
    return () => clearInterval(interval)
  }, [])

  const handleScrapeResults = async () => {
    setScrapingResults(true); setScrapeMessage(null)
    try {
      const result = await api.scrapeResults()
      setScrapeMessage(result.message || `Capturé ${result.results_count} résultats`)
      loadDashboard()
    } catch (error: any) {
      setScrapeMessage('Erreur: ' + (error.response?.data?.detail || error.message))
    } finally { setScrapingResults(false) }
  }

  const handleScrapeMatches = async () => {
    setScrapingMatches(true); setScrapeMessage(null)
    try {
      const result = await api.scrapeMatches()
      setScrapeMessage(result.message || `Capturé ${result.matches_count} matchs`)
      loadDashboard()
    } catch (error: any) {
      setScrapeMessage('Erreur: ' + (error.response?.data?.detail || error.message))
    } finally { setScrapingMatches(false) }
  }

  const handleNewSeason = async () => {
    if (!window.confirm('Créer une nouvelle saison ?')) return
    setCreatingSeason(true); setScrapeMessage(null)
    try {
      const result = await api.createNewSeason()
      setScrapeMessage(result.message || `Saison ${result.season_number} créée`)
      loadDashboard()
    } catch (error: any) {
      setScrapeMessage('Erreur: ' + (error.response?.data?.detail || error.message))
    } finally { setCreatingSeason(false) }
  }

  if (loading) return <div className="text-center py-10 flex flex-col items-center justify-center"><Loader2 className="w-8 h-8 animate-spin text-blue-500 mb-4" /> Chargement du Centre de Commande...</div>

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        {/* COMMENTÉ - Titre Elite 5
        <div>
          <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-amber-400 to-yellow-500">
            🏆 Elite 5 — Prédictions Ultra-Sûres
          </h1>
          <div className="flex items-center space-x-4 mt-2">
            <p className="text-gray-400">5 paris par saison • Cote &gt; 2.00 • Quasi-certitude</p>
        */}
        {/* FIN COMMENTAIRE - Titre Elite 5 */}
        <div>
          <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
            📊 Centre de Commande
          </h1>
          <div className="flex items-center space-x-4 mt-2">
            <p className="text-gray-400">Tableau de bord prédictions</p>
            {overview?.season?.progress && (
              <span className="px-3 py-1 bg-gray-800 border border-gray-700 rounded-full text-sm text-gray-300 font-medium flex items-center">
                <Calendar className="w-4 h-4 mr-2 text-blue-400" />
                J{overview.season.progress.completed_matchdays} / {overview.season.progress.total_matchdays} 
                <span className="text-gray-500 ml-2">({overview.season.progress.progress_percent.toFixed(0)}%)</span>
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center space-x-3">
          <button onClick={handleScrapeResults} disabled={scrapingResults} className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-500 rounded-lg transition-colors">
            {scrapingResults ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
            <span>Sync Résultats</span>
          </button>
          <button onClick={handleScrapeMatches} disabled={scrapingMatches} className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors">
            {scrapingMatches ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
            <span>Sync Matchs</span>
          </button>
          <button onClick={handleNewSeason} disabled={creatingSeason} className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-500 rounded-lg transition-colors">
            {creatingSeason ? <Loader2 className="w-4 h-4 animate-spin" /> : <PlusCircle className="w-4 h-4" />}
            <span>Saison++</span>
          </button>
        </div>
      </div>

      {scrapeMessage && (
        <div className={`px-4 py-3 rounded-lg border-l-4 ${scrapeMessage.includes('Erreur') ? 'bg-red-900/50 border-red-500 text-red-100' : 'bg-green-900/50 border-green-500 text-green-100'}`}>
          {scrapeMessage}
        </div>
      )}

      {/* ═══════════ ELITE SLOTS TRACKER ═══════════ */}
      {/* COMMENTÉ - Slots Elite
      <div className="bg-gradient-to-r from-amber-900/30 via-yellow-900/20 to-amber-900/30 backdrop-blur border border-amber-700/50 p-6 rounded-xl shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold flex items-center text-amber-300">
            <Trophy className="w-6 h-6 mr-2" />
            Slots Elite — Saison {overview?.season?.number}
          </h2>
          <div className="flex items-center space-x-2">
            <span className="text-3xl font-black text-white">{slotsUsed}</span>
            <span className="text-gray-400 text-lg">/</span>
            <span className="text-xl font-bold text-gray-400">{slotsMax}</span>
            <span className="text-sm text-gray-500 ml-2">utilisés</span>
          </div>
        </div>
        
        <div className="grid grid-cols-5 gap-3 mb-4">
          {[0, 1, 2, 3, 4].map((idx) => {
            const ep = elite?.elite_predictions?.[idx]
            const isUsed = !!ep
            const isPending = isUsed && ep.actual_result === null
            const isCorrect = isUsed && ep.is_correct === true
            const isWrong = isUsed && ep.is_correct === false

            return (
              <div 
                key={idx}
                className={`p-3 rounded-lg border-2 transition-all ${
                  isCorrect ? 'bg-green-900/40 border-green-500 shadow-green-500/20 shadow-lg' :
                  isWrong ? 'bg-red-900/40 border-red-500 shadow-red-500/20 shadow-lg' :
                  isPending ? 'bg-amber-900/30 border-amber-500 animate-pulse shadow-amber-500/20 shadow-lg' :
                  'bg-gray-800/50 border-gray-700 border-dashed'
                }`}
              >
                <div className="text-center">
                  <div className="text-xs text-gray-500 mb-1 font-medium">Slot #{idx + 1}</div>
                  {isUsed ? (
                    <>
                      <div className="text-xs font-bold text-white truncate" title={`${ep.home_team} vs ${ep.away_team}`}>
                        {ep.home_team?.split(' ')[0]} vs {ep.away_team?.split(' ')[0]}
                      </div>
                      <div className="mt-1">
                        <span className={`inline-block px-2 py-0.5 rounded text-xs font-black text-white ${
                          ep.predicted_result === 'V' ? 'bg-green-600' : 
                          ep.predicted_result === 'N' ? 'bg-yellow-600' : 'bg-red-600'
                        }`}>
                          {ep.predicted_result === 'V' ? '1' : ep.predicted_result === 'N' ? 'X' : '2'} @{ep.odds?.toFixed(2)}
                        </span>
                      </div>
                      <div className="text-xs text-gray-400 mt-1">J{ep.matchday}</div>
                      {isPending && <div className="text-xs text-amber-400 mt-1 font-bold">⏳ En attente</div>}
                      {isCorrect && <div className="text-xs text-green-400 mt-1 font-bold">✅ +{ep.profit_loss?.toFixed(0)} Ar</div>}
                      {isWrong && <div className="text-xs text-red-400 mt-1 font-bold">❌ {ep.profit_loss?.toFixed(0)} Ar</div>}
                    </>
                  ) : (
                    <>
                      <Lock className="w-5 h-5 mx-auto text-gray-600 my-1" />
                      <div className="text-xs text-gray-600">Disponible</div>
                    </>
                  )}
                </div>
              </div>
            )
          })}
        </div>
        
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center space-x-4">
            <span className="text-gray-400">
              {elite?.candidates_rejected || 0} matchs analysés et rejetés ce saison
            </span>
          </div>
          <div className="flex items-center space-x-3">
            {elite?.verified_count > 0 && (
              <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                (elite.accuracy || 0) >= 0.6 ? 'bg-green-900/50 border border-green-600 text-green-300' : 'bg-red-900/50 border border-red-600 text-red-300'
              }`}>
                Précision: {((elite.accuracy || 0) * 100).toFixed(0)}%
              </span>
            )}
            {elite?.total_profit !== undefined && elite?.total_profit !== 0 && (
              <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                elite.total_profit >= 0 ? 'bg-green-900/50 border border-green-600 text-green-300' : 'bg-red-900/50 border border-red-600 text-red-300'
              }`}>
                P/L: {elite.total_profit >= 0 ? '+' : ''}{elite.total_profit?.toFixed(0)} Ar
              </span>
            )}
          </div>
        </div>
      </div>
      */}
      {/* FIN COMMENTAIRE - Slots Elite */}

      {/* ═══════════ MAIN CONTENT ═══════════ */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Colonne 1: Matchs à Venir */}
        <div className="bg-gray-800/80 backdrop-blur border border-gray-700 p-5 rounded-xl">
          <h2 className="text-lg font-bold mb-4 flex items-center">
            <Target className="w-5 h-5 mr-2 text-yellow-400" />
            Matchs à Venir ({overview?.upcoming_matches?.length || 0})
          </h2>
          <div className="space-y-3">
            {(!overview?.upcoming_matches || overview.upcoming_matches.length === 0) ? (
              <p className="text-gray-500 text-center py-4">Aucun match, cliquez sur Sync Matchs.</p>
            ) : (
              overview.upcoming_matches.slice(0, 10).map((match: any) => (
                <div key={match.id} className="flex justify-between items-center bg-gray-900/50 p-3 rounded-lg border border-gray-700/50 hover:bg-gray-750 transition-colors">
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="font-medium text-gray-300">{match.home_team}</span>
                      <span className="text-gray-600 text-xs">vs</span>
                      <span className="font-medium text-gray-300">{match.away_team}</span>
                    </div>
                    <span className="text-xs text-gray-500">J{match.matchday}</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="px-2 py-1 rounded bg-gray-800 border border-gray-700 text-xs font-mono text-gray-300"><span className="text-gray-500 mr-1">1</span>{match.odds?.home?.toFixed(2)}</span>
                    <span className="px-2 py-1 rounded bg-gray-800 border border-gray-700 text-xs font-mono text-gray-300"><span className="text-gray-500 mr-1">X</span>{match.odds?.draw?.toFixed(2)}</span>
                    <span className="px-2 py-1 rounded bg-gray-800 border border-gray-700 text-xs font-mono text-gray-300"><span className="text-gray-500 mr-1">2</span>{match.odds?.away?.toFixed(2)}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Colonne 2: Analyse du Moteur (toutes les prédictions — info seulement) */}
        <div className="bg-gray-800/80 backdrop-blur border border-gray-700 p-5 rounded-xl">
          <h2 className="text-lg font-bold mb-4 flex items-center">
            <Brain className="w-5 h-5 mr-2 text-blue-400" />
            Analyse Moteur ({overview?.all_predictions?.length || 0})
            <span className="ml-2 text-xs text-gray-500 font-normal">(info seulement)</span>
          </h2>
          <div className="space-y-3">
            {(!overview?.all_predictions || overview.all_predictions.length === 0) ? (
              <p className="text-gray-500 text-center py-4">Aucune analyse en cours.</p>
            ) : (
              overview.all_predictions.slice(0, 10).map((match: any) => (
                <div key={match.id} className={`flex justify-between items-center p-3 rounded-lg border transition-colors ${
                  match.is_elite 
                    ? 'bg-amber-900/20 border-amber-600/50' 
                    : 'bg-gray-900/50 border-gray-700/50'
                }`}>
                  <div>
                    <div className="flex items-center space-x-2">
                      {match.is_elite && <span className="text-amber-400">★</span>}
                      <span className="font-medium text-gray-300">{match.home_team}</span>
                      <span className="text-gray-600 text-xs">vs</span>
                      <span className="font-medium text-gray-300">{match.away_team}</span>
                    </div>
                    <div className="flex items-center space-x-2 mt-1">
                      <span className="text-xs text-gray-500">Conf: {(match.confidence * 100).toFixed(1)}%</span>
                      <span className="text-xs text-gray-500">•</span>
                      <span className="text-xs text-gray-500">Agree: {((match.model_agreement || 0) * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                  <div className="flex flex-col items-end">
                    <span className={`px-3 py-1 rounded text-sm font-black text-white ${match.predicted_result === 'V' ? 'bg-green-600/60' : match.predicted_result === 'N' ? 'bg-yellow-600/60' : 'bg-red-600/60'}`}>
                      {match.predicted_result === 'V' ? '1' : match.predicted_result === 'N' ? 'X' : '2'}
                    </span>
                    {match.is_elite && (
                      <span className="text-xs text-amber-400 font-bold mt-1">🏆 ELITE</span>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Colonne 3: Derniers Résultats */}
        <div className="bg-gray-800/80 backdrop-blur border border-gray-700 p-5 rounded-xl">
          <h2 className="text-lg font-bold mb-4 flex items-center">
            <Activity className="w-5 h-5 mr-2 text-blue-400" />
            Vérité Terrain (Derniers Résultats)
          </h2>
          <div className="space-y-3">
            {(!overview?.recent_results || overview.recent_results.length === 0) ? (
              <p className="text-gray-500 text-center py-4">Aucun résultat récent</p>
            ) : (
              overview.recent_results.slice(0, 10).map((match: any) => (
                <div key={match.id} className="flex justify-between items-center bg-gray-900/50 p-3 rounded-lg border border-gray-700/50 hover:bg-gray-750 transition-colors">
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="font-medium text-gray-300">{match.home_team}</span>
                      <span className="text-gray-600 text-xs">vs</span>
                      <span className="font-medium text-gray-300">{match.away_team}</span>
                    </div>
                    <span className="text-xs text-gray-500">Matchday {match.matchday}</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <span className="font-black text-lg text-white">{match.score}</span>
                    <span className={`w-8 h-8 flex items-center justify-center rounded font-black text-white ${match.result === 'V' ? 'bg-green-600/80' : match.result === 'N' ? 'bg-yellow-600/80' : 'bg-red-600/80'}`}>
                      {match.result === 'V' ? '1' : match.result === 'N' ? 'X' : '2'}
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* ═══════════ REGIME & SIGNALS ═══════════ */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* État du Système */}
        <div className="bg-gray-800/80 backdrop-blur border border-gray-700 p-5 rounded-xl shadow-lg relative overflow-hidden">
          <div className="absolute shrink-0 top-0 right-0 w-16 h-16 bg-red-500/10 rounded-bl-full -mr-4 -mt-4"></div>
          <div className="flex items-center space-x-3 mb-3">
            <div className="p-2 bg-red-500/20 rounded-lg text-red-400"><ShieldAlert className="w-5 h-5" /></div>
            <span className="text-gray-400 font-medium">ÉTAT SYSTÈME</span>
          </div>
          <div className="text-3xl font-black text-white mb-1">
            {rtData?.engine_state?.regime || 'STABLE'}
          </div>
          <div className={`text-sm font-bold ${rtData?.learning?.recovery_mode ? 'text-red-500 animate-pulse' : 'text-green-500'}`}>
            {rtData?.learning?.recovery_mode ? '🚨 MODE RÉCUPÉRATION' : '✓ SYSTÈME NOMINAL'}
          </div>
        </div>

        {/* Régime & Signaux */}
        <div className="bg-gray-800/80 backdrop-blur border border-gray-700 p-5 rounded-xl shadow-lg">
          <h2 className="text-lg font-bold flex items-center mb-4">
            <Crosshair className="w-5 h-5 mr-2 text-purple-400" />
            Régime Engine & Signaux
          </h2>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Régime</span>
                <span className={`font-bold ${rtData?.engine_state?.regime === 'CHAOTIC' ? 'text-red-400' : 'text-green-400'}`}>{rtData?.engine_state?.regime || 'STABLE'}</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className={`h-2 rounded-full ${rtData?.engine_state?.regime === 'CHAOTIC' ? 'bg-red-500 w-full' : rtData?.engine_state?.regime === 'TRANSITION' ? 'bg-yellow-500 w-2/3' : 'bg-green-500 w-1/3'}`}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">KL Div.</span>
                <span className="font-mono text-blue-400">{rtData?.engine_state?.kl_divergence?.toFixed(3) || '0.000'}</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className="bg-blue-500 h-2 rounded-full transition-all" style={{ width: `${Math.min(((rtData?.engine_state?.kl_divergence || 0) / 0.1) * 100, 100)}%` }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">P-Value</span>
                <span className="font-mono text-purple-400">{rtData?.engine_state?.runs_pvalue?.toFixed(3) || '0.500'}</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className="bg-purple-500 h-2 rounded-full transition-all" style={{ width: `${Math.min((rtData?.engine_state?.runs_pvalue || 0) * 100, 100)}%` }}></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
