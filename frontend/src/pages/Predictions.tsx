import { useState, useEffect } from 'react'
import { Brain, X, TrendingUp, TrendingDown, Minus, DollarSign, BarChart3, ChevronLeft, ChevronRight, CheckCircle, XCircle, Cpu } from 'lucide-react'
import api from '../services/api'
import { useRefreshOnUpdate, useSeasonReset } from '../hooks/useRealTimeUpdates'
import PredictionComparisonModal from '../components/PredictionComparisonModal'
import EngineSignalsPanel from '../components/EngineSignalsPanel'

// Fixed stake in Ariary
const FIXED_STAKE_ARIARY = 1000

// Format currency in Ariary
const formatAriary = (amount: number) => {
  return new Intl.NumberFormat('mg-MG', {
    style: 'decimal',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(amount) + ' Ar'
}

interface Prediction {
  id: number
  match_id: number
  predicted_result: string
  predicted_result_name: string
  prob_home_win: number
  prob_draw: number
  prob_away_win: number
  confidence: number
  model_agreement: number
  value_home: number | null
  value_draw: number | null
  value_away: number | null
  best_value_outcome: string | null
  best_value_amount: number | null
  actual_result: string | null
  is_correct: boolean | null
  verified_at: string | null
  // Selection fields - CRITICAL for backend/frontend sync
  is_selected_for_bet: boolean
  selection_rank: number | null
  selection_reason: string | null
}

interface Match {
  id: number
  matchday: number
  home_team: string
  away_team: string
  odd_home: number
  odd_draw: number
  odd_away: number
  score_home: number | null
  score_away: number | null
  result: string | null
  is_completed: boolean
}

interface PredictionWithMatch {
  prediction: Prediction
  match: Match
}

interface BetResult {
  bet_outcome: string | null
  bet_odds: number | null
  stake: number | null
  profit_loss: number | null
  is_win: boolean | null
}

export default function Predictions() {
  const [predictions, setPredictions] = useState<PredictionWithMatch[]>([])
  const [accuracy, setAccuracy] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [selectedPrediction, setSelectedPrediction] = useState<PredictionWithMatch | null>(null)
  const [betResult, setBetResult] = useState<BetResult | null>(null)
  const [verifiedMatchdays, setVerifiedMatchdays] = useState<number[]>([])
  const [comparisonData, setComparisonData] = useState<any>(null)
  const [showResultsModal, setShowResultsModal] = useState(false)
  const [loadingResults, setLoadingResults] = useState(false)
  
  // Results navigation state
  const [resultsMatchday, setResultsMatchday] = useState<number | null>(null)
  const [resultsData, setResultsData] = useState<any>(null)
  const [loadingResultsMatchday, setLoadingResultsMatchday] = useState(false)
  
  // Season history state
  const [seasonHistory, setSeasonHistory] = useState<any[]>([])
  const [loadingHistory, setLoadingHistory] = useState(false)
  const [currentPage, setCurrentPage] = useState(1)
  const seasonsPerPage = 10

  const loadPredictions = async () => {
    try {
      // Parallelise ALL API calls for faster loading
      const [predData, accData, historyData, matchdaysData] = await Promise.all([
        api.getUpcomingPredictions(),
        api.getPredictionAccuracy(),
        api.getSeasonBettingHistory(),
        api.getVerifiedMatchdays()
      ])
      
      const typedPredData = predData as PredictionWithMatch[]
      setPredictions(typedPredData)
      setAccuracy(accData)
      setSeasonHistory(historyData.seasons || [])
      setVerifiedMatchdays(matchdaysData.matchdays || [])
      
      // Set initial results matchday to the most recent
      if (matchdaysData.matchdays && matchdaysData.matchdays.length > 0 && resultsMatchday === null) {
        setResultsMatchday(matchdaysData.matchdays[0])
      }
    } catch (error) {
      console.error('Failed to load predictions:', error)
    } finally {
      setLoading(false)
    }
  }

  const resetAllData = () => {
    console.log('resetAllData called - resetting all prediction data');
    // Reset all state to initial values
    setPredictions([])
    setAccuracy(null)
    setSelectedPrediction(null)
    setBetResult(null)
    setVerifiedMatchdays([])
    setComparisonData(null)
    setShowResultsModal(false)
    setResultsMatchday(null)
    setResultsData(null)
    setSeasonHistory([])
    setCurrentPage(1)
    setLoading(true)
    setLoadingHistory(false)
    
    console.log('All state reset, reloading fresh data...');
    // Reload fresh data for new season (single call - loadPredictions handles all)
    setTimeout(() => {
      loadPredictions()
    }, 100)
  }

  useRefreshOnUpdate(loadPredictions)
  useSeasonReset(resetAllData)

  useEffect(() => {
    loadPredictions()
  }, [])

    
  const loadResultsMatchday = async (matchday: number) => {
    setLoadingResultsMatchday(true)
    try {
      const data = await api.getMatchdayComparison(matchday)
      setResultsData(data)
    } catch (error) {
      console.error('Failed to load results:', error)
      setResultsData(null)
    } finally {
      setLoadingResultsMatchday(false)
    }
  }
  
  useEffect(() => {
    if (resultsMatchday !== null) {
      loadResultsMatchday(resultsMatchday)
    }
  }, [resultsMatchday])

  const showAllResults = async () => {
    setLoadingResults(true)
    try {
      const data = await api.getAllComparisons()
      if (data && data.predictions && data.predictions.length > 0) {
        setComparisonData(data)
        setShowResultsModal(true)
      } else {
        alert('Aucun résultat vérifié disponible')
      }
    } catch (error) {
      console.error('Failed to load all results:', error)
    } finally {
      setLoadingResults(false)
    }
  }

  const openPredictionModal = async (pred: PredictionWithMatch) => {
    setSelectedPrediction(pred)
    // Fetch bet result if prediction was verified
    if (pred.prediction.verified_at && pred.match.is_completed) {
      try {
        const betData = await api.getBetForMatch(pred.match.id)
        setBetResult(betData)
      } catch {
        setBetResult(null)
      }
    } else {
      setBetResult(null)
    }
  }

  const filteredPredictions = predictions.slice(0, 10)

  const getResultIcon = (result: string | null) => {
    if (!result) return <Minus className="w-5 h-5 text-gray-400" />
    if (result === 'V') return <TrendingUp className="w-5 h-5 text-green-500" />
    if (result === 'N') return <Minus className="w-5 h-5 text-yellow-500" />
    return <TrendingDown className="w-5 h-5 text-red-500" />
  }

  const getResultName = (result: string | null) => {
    if (!result) return '-'
    if (result === 'V') return 'Victoire Domicile (1)'
    if (result === 'N') return 'Match Nul (X)'
    return 'Victoire Extérieur (2)'
  }

  if (loading) {
    return <div className="text-center py-10">Loading predictions...</div>
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Predictions</h1>

      {/* Real-Time Engine Signals Panel */}
      <div className="bg-gray-800/30 border border-gray-700/60 rounded-xl p-5">
        <div className="flex items-center gap-2 mb-4">
          <Cpu className="w-5 h-5 text-purple-400" />
          <h2 className="text-lg font-bold text-gray-100">Moteur Statistique Temps Réel</h2>
          <span className="text-xs text-gray-500 ml-auto">Signaux M1-M15 • Mise à jour auto 30s</span>
        </div>
        <EngineSignalsPanel />
      </div>

      {/* Mise fixe info */}
      <div className="bg-blue-900/30 border border-blue-500 rounded-lg p-4 mb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <DollarSign className="w-5 h-5 text-blue-400" />
            <span className="text-blue-300 font-medium">Mise fixe par prédiction: </span>
            <span className="text-white font-bold">{formatAriary(FIXED_STAKE_ARIARY)}</span>
          </div>
          
          {/* Button to view results */}
          {verifiedMatchdays.length > 0 && (
            <button
              onClick={showAllResults}
              disabled={loadingResults}
              className="flex items-center space-x-2 bg-green-600 hover:bg-green-500 disabled:bg-gray-600 px-4 py-2 rounded-lg text-white font-medium transition-colors"
            >
              <BarChart3 className="w-5 h-5" />
              <span>{loadingResults ? 'Chargement...' : 'Voir les résultats'}</span>
            </button>
          )}
        </div>
      </div>
      
      {/* Results by Matchday Section */}
      {verifiedMatchdays.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-4 mb-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <BarChart3 className="w-5 h-5 text-green-400" />
              <span className="text-gray-300 font-medium">Résultats par journée</span>
            </div>
            
            {/* Navigation */}
            {resultsMatchday !== null && (
              <div className="flex items-center space-x-3">
                <button 
                  disabled={verifiedMatchdays.indexOf(resultsMatchday) >= verifiedMatchdays.length - 1}
                  onClick={() => {
                    const idx = verifiedMatchdays.indexOf(resultsMatchday)
                    if (idx < verifiedMatchdays.length - 1) setResultsMatchday(verifiedMatchdays[idx + 1])
                  }}
                  className="flex items-center px-3 py-1.5 bg-gray-700 rounded-lg hover:bg-gray-600 disabled:opacity-50 text-sm"
                  title="Journée plus récente"
                >
                  <ChevronLeft className="w-4 h-4 mr-1" />
                  Plus récent
                </button>
                <span className="font-bold text-lg px-3 py-1.5 bg-gray-900 rounded">
                  J{resultsMatchday}
                </span>
                <button 
                  disabled={verifiedMatchdays.indexOf(resultsMatchday) <= 0}
                  onClick={() => {
                    const idx = verifiedMatchdays.indexOf(resultsMatchday)
                    if (idx > 0) setResultsMatchday(verifiedMatchdays[idx - 1])
                  }}
                  className="flex items-center px-3 py-1.5 bg-blue-600 rounded-lg hover:bg-blue-500 disabled:opacity-50 text-sm"
                  title="Journée plus ancienne"
                >
                  Plus ancien
                  <ChevronRight className="w-4 h-4 ml-1" />
                </button>
              </div>
            )}
          </div>
          
          {/* Results Content */}
          {loadingResultsMatchday ? (
            <div className="text-center py-8 text-gray-400">Chargement...</div>
          ) : resultsData && resultsData.predictions && resultsData.predictions.length > 0 ? (
            <div>
              {/* Summary */}
              <div className="grid grid-cols-4 gap-3 mb-4">
                <div className="bg-gray-700 rounded-lg p-3 text-center">
                  <div className="text-gray-400 text-xs">Total Misé</div>
                  <div className="text-lg font-bold">{formatAriary(resultsData.total_stake || 0)}</div>
                </div>
                <div className="bg-gray-700 rounded-lg p-3 text-center">
                  <div className="text-gray-400 text-xs">Gain/Perte</div>
                  <div className={`text-lg font-bold ${(resultsData.total_profit || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {(resultsData.total_profit || 0) >= 0 ? '+' : ''}{formatAriary(resultsData.total_profit || 0)}
                  </div>
                </div>
                <div className="bg-gray-700 rounded-lg p-3 text-center">
                  <div className="text-gray-400 text-xs">Précision</div>
                  <div className="text-lg font-bold text-blue-400">
                    {((resultsData.accuracy || 0) * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="bg-gray-700 rounded-lg p-3 text-center">
                  <div className="text-gray-400 text-xs">Correctes</div>
                  <div className="text-lg font-bold">
                    {resultsData.correct_count || 0}/{resultsData.total_predictions || 0}
                  </div>
                </div>
              </div>
              
              {/* Predictions Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="text-gray-400 text-xs bg-gray-750">
                    <tr>
                      <th className="p-2 text-left">Match</th>
                      <th className="p-2 text-center">Prédit</th>
                      <th className="p-2 text-center">Résultat</th>
                      <th className="p-2 text-center">Score</th>
                      <th className="p-2 text-right">P/L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {resultsData.predictions.map((pred: any, idx: number) => (
                      <tr 
                        key={idx} 
                        className={`border-t border-gray-700/50 ${pred.is_correct ? 'bg-green-900/10' : 'bg-red-900/10'}`}
                      >
                        <td className="p-2 font-medium">{pred.match}</td>
                        <td className="p-2 text-center">
                          <span className={`px-2 py-1 rounded text-xs font-bold ${
                            pred.predicted === 'Home Win' ? 'bg-green-600' :
                            pred.predicted === 'Draw' ? 'bg-yellow-600' : 'bg-red-600'
                          }`}>
                            {pred.predicted === 'Home Win' ? '1' : pred.predicted === 'Draw' ? 'X' : '2'}
                          </span>
                        </td>
                        <td className="p-2 text-center">
                          <span className={`px-2 py-1 rounded text-xs font-bold ${
                            pred.actual_result === 'V' ? 'bg-green-600' :
                            pred.actual_result === 'N' ? 'bg-yellow-600' : 'bg-red-600'
                          }`}>
                            {pred.actual_result === 'V' ? '1' : pred.actual_result === 'N' ? 'X' : '2'}
                          </span>
                        </td>
                        <td className="p-2 text-center">{pred.score || '-'}</td>
                        <td className="p-2 text-right">
                          <div className="flex items-center justify-end space-x-1">
                            {pred.is_correct ? (
                              <CheckCircle className="w-4 h-4 text-green-400" />
                            ) : (
                              <XCircle className="w-4 h-4 text-red-400" />
                            )}
                            <span className={`font-bold ${pred.profit_loss >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {pred.profit_loss >= 0 ? '+' : ''}{formatAriary(pred.profit_loss)}
                            </span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-400">Aucun résultat pour cette journée</div>
          )}
        </div>
      )}

      {/* Accuracy Stats */}
      {accuracy && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="card">
            <div className="text-gray-400 text-sm">Total Predictions</div>
            <div className="text-2xl font-bold">{accuracy.total_predictions}</div>
          </div>
          <div className="card">
            <div className="text-gray-400 text-sm">Accuracy</div>
            <div className="text-2xl font-bold text-green-500">
              {(accuracy.accuracy * 100).toFixed(1)}%
            </div>
          </div>
          <div className="card">
            <div className="text-gray-400 text-sm">Home Win Accuracy</div>
            <div className="text-2xl font-bold">
              {accuracy.home_win_predictions > 0 
                ? ((accuracy.home_win_correct / accuracy.home_win_predictions) * 100).toFixed(1)
                : 0}%
            </div>
          </div>
          <div className="card">
            <div className="text-gray-400 text-sm">Avg Confidence</div>
            <div className="text-2xl font-bold">
              {(accuracy.avg_confidence * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      )}

      {/* Season History */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold flex items-center">
            <DollarSign className="w-5 h-5 mr-2 text-yellow-400" />
            Historique par Saison
          </h2>
          {seasonHistory.length > 0 && (
            <div className="text-sm text-gray-400">
              {seasonHistory.length} saison{seasonHistory.length > 1 ? 's' : ''}
            </div>
          )}
        </div>
        
        {loadingHistory ? (
          <div className="text-center py-8 text-gray-400">Chargement de l'historique...</div>
        ) : seasonHistory.length === 0 ? (
          <div className="text-center py-8 text-gray-400">Aucune saison trouvée</div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-gray-400 text-xs bg-gray-750">
                  <tr>
                    <th className="p-2 text-left">Saison</th>
                    <th className="p-2 text-center">Statut</th>
                    <th className="p-2 text-center">Total Misé</th>
                    <th className="p-2 text-center">Gain/Perte</th>
                    <th className="p-2 text-center">ROI</th>
                    <th className="p-2 text-center">Précision</th>
                    <th className="p-2 text-center">Paris</th>
                  </tr>
                </thead>
                <tbody>
                  {seasonHistory
                    .slice((currentPage - 1) * seasonsPerPage, currentPage * seasonsPerPage)
                    .map((season) => (
                      <tr key={season.season_number} className="border-b border-gray-700 hover:bg-gray-750">
                        <td className="p-2 font-medium">
                          Saison {season.season_number}
                          {season.is_active && !season.is_completed && <span className="ml-2 text-xs bg-green-600 px-2 py-1 rounded">Active</span>}
                        </td>
                        <td className="p-2 text-center">
                          <span className={season.status === 'En cours' ? 'text-green-400' : season.status === 'Terminée' ? 'text-gray-400' : 'text-yellow-400'}>
                            {season.status || (season.is_active ? 'En cours' : 'Terminée')}
                          </span>
                        </td>
                        <td className="p-2 text-center">{formatAriary(season.total_stake || 0)}</td>
                        <td className={`p-2 text-center font-bold ${(season.total_profit || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {(season.total_profit || 0) >= 0 ? '+' : ''}{formatAriary(season.total_profit || 0)}
                        </td>
                        <td className={`p-2 text-center ${(season.roi || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {((season.roi || 0) * 100).toFixed(1)}%
                        </td>
                        <td className="p-2 text-center text-blue-400">
                          {((season.win_rate || 0) * 100).toFixed(1)}%
                        </td>
                        <td className="p-2 text-center text-gray-400">
                          {season.winning_bets || 0}/{season.total_bets || 0}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
            
            {/* Pagination */}
            {seasonHistory.length > seasonsPerPage && (
              <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-700">
                <div className="text-sm text-gray-400">
                  Affichage {((currentPage - 1) * seasonsPerPage) + 1} - {Math.min(currentPage * seasonsPerPage, seasonHistory.length)} sur {seasonHistory.length} saisons
                </div>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                    disabled={currentPage === 1}
                    className="flex items-center space-x-1 px-3 py-1.5 bg-gray-700 rounded-lg hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
                  >
                    <ChevronLeft className="w-4 h-4" />
                    <span>Précédent</span>
                  </button>
                  
                  <span className="px-3 py-1.5 bg-gray-900 rounded-lg text-sm font-medium">
                    {currentPage} / {Math.ceil(seasonHistory.length / seasonsPerPage)}
                  </span>
                  
                  <button
                    onClick={() => setCurrentPage(prev => Math.min(Math.ceil(seasonHistory.length / seasonsPerPage), prev + 1))}
                    disabled={currentPage === Math.ceil(seasonHistory.length / seasonsPerPage)}
                    className="flex items-center space-x-1 px-3 py-1.5 bg-blue-600 rounded-lg hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
                  >
                    <span>Suivant</span>
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Upcoming Predictions */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold flex items-center">
            <Brain className="w-5 h-5 mr-2" />
            Sélections du jour
          </h2>
          
          {filteredPredictions.length > 0 && (
            <div className="bg-green-900/50 border border-green-500 rounded-lg px-4 py-2">
              <span className="text-green-300 text-sm">Prédictions</span>
              <span className="text-white font-bold text-xl ml-2">{filteredPredictions.length}</span>
            </div>
          )}
        </div>
        
        {filteredPredictions.length === 0 ? (
          <div className="text-center py-8">
            <p className="text-gray-400">Aucune prédiction pour cette journée</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-400 text-sm border-b border-gray-700">
                  <th className="pb-2">#</th>
                  <th className="pb-2 min-w-[300px]">Match</th>
                  <th className="pb-2">Prédiction</th>
                  <th className="pb-2">Cotes</th>
                </tr>
              </thead>
              <tbody>
                {filteredPredictions.map(({ prediction, match }, index) => (
                  <tr 
                    key={prediction.id} 
                    className="border-b border-gray-700 hover:bg-gray-750 cursor-pointer"
                    onClick={() => openPredictionModal({ prediction, match })}
                  >
                    <td className="py-3">
                      <span className="bg-green-600 text-white px-2 py-1 rounded font-bold text-sm">
                        #{prediction.selection_rank || index + 1}
                      </span>
                    </td>
                    <td className="py-3">
                      <div className="flex items-center space-x-2">
                        <div>
                          <span className="font-bold text-lg text-gray-200">{match.home_team}</span>
                          <span className="text-gray-500 mx-3 text-sm">vs</span>
                          <span className="font-bold text-lg text-gray-200">{match.away_team}</span>
                        </div>
                      </div>
                    </td>
                    <td className="py-3">
                      <span className={`px-4 py-2 rounded font-black text-white shadow-lg ${
                        prediction.predicted_result === 'V' ? 'bg-green-600' :
                        prediction.predicted_result === 'N' ? 'bg-yellow-600' : 'bg-red-600'
                      }`}>
                        {prediction.predicted_result_name}
                      </span>
                    </td>
                    <td className="py-3">
                      <div className="flex space-x-2">
                        <span className="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm font-mono text-gray-300">
                          <span className="text-gray-500 mr-1">1</span>{match.odd_home?.toFixed(2)}
                        </span>
                        <span className="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm font-mono text-gray-300">
                          <span className="text-gray-500 mr-1">X</span>{match.odd_draw?.toFixed(2)}
                        </span>
                        <span className="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm font-mono text-gray-300">
                          <span className="text-gray-500 mr-1">2</span>{match.odd_away?.toFixed(2)}
                        </span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Prediction Detail Modal */}
      {selectedPrediction && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setSelectedPrediction(null)}>
          <div className="bg-gray-800 rounded-lg p-6 max-w-lg w-full mx-4" onClick={e => e.stopPropagation()}>
            <div className="flex justify-between items-start mb-4">
              <h3 className="text-xl font-bold">Détail Prédiction</h3>
              <button onClick={() => setSelectedPrediction(null)} className="text-gray-400 hover:text-white">
                <X className="w-6 h-6" />
              </button>
            </div>

            <div className="space-y-4">
              {/* Match Info */}
              <div className="bg-gray-700 rounded p-4">
                <div className="text-lg font-semibold text-center">
                  {selectedPrediction.match.home_team} vs {selectedPrediction.match.away_team}
                </div>
                <div className="text-gray-400 text-center text-sm">Journée {selectedPrediction.match.matchday}</div>
              </div>

              {/* Prediction vs Result */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-700 rounded p-4">
                  <div className="text-gray-400 text-sm mb-1">Prédiction</div>
                  <div className="flex items-center space-x-2">
                    {getResultIcon(selectedPrediction.prediction.predicted_result)}
                    <span className="font-bold">{selectedPrediction.prediction.predicted_result_name}</span>
                  </div>
                  <div className="text-sm text-gray-400 mt-2">
                    Confiance: {(selectedPrediction.prediction.confidence * 100).toFixed(0)}%
                  </div>
                </div>
                
                <div className="bg-gray-700 rounded p-4">
                  <div className="text-gray-400 text-sm mb-1">Résultat Réel</div>
                  {selectedPrediction.match.is_completed ? (
                    <>
                      <div className="flex items-center space-x-2">
                        {getResultIcon(selectedPrediction.match.result)}
                        <span className="font-bold">{getResultName(selectedPrediction.match.result)}</span>
                      </div>
                      <div className="text-sm mt-2">
                        Score: <span className="font-bold">{selectedPrediction.match.score_home} - {selectedPrediction.match.score_away}</span>
                      </div>
                    </>
                  ) : (
                    <span className="text-gray-500">En attente du match</span>
                  )}
                </div>
              </div>

              {/* Comparison Result */}
              {selectedPrediction.match.is_completed && (
                <div className={`rounded p-4 ${selectedPrediction.prediction.is_correct ? 'bg-green-900/50' : 'bg-red-900/50'}`}>
                  <div className="flex items-center justify-between">
                    <span className="font-semibold">
                      {selectedPrediction.prediction.is_correct ? '✓ Prédiction Correcte!' : '✗ Prédiction Incorrecte'}
                    </span>
                    {selectedPrediction.prediction.is_correct ? (
                      <TrendingUp className="w-6 h-6 text-green-500" />
                    ) : (
                      <TrendingDown className="w-6 h-6 text-red-500" />
                    )}
                  </div>
                </div>
              )}

              {/* Bet Result */}
              {betResult && betResult.stake && (
                <div className="bg-gray-700 rounded p-4">
                  <div className="flex items-center space-x-2 mb-3">
                    <DollarSign className="w-5 h-5 text-yellow-500" />
                    <span className="font-semibold">Résultat du Pari</span>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-gray-400">Pari:</span>
                      <span className="ml-2 font-medium">{betResult.bet_outcome}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Cote:</span>
                      <span className="ml-2 font-medium">{betResult.bet_odds?.toFixed(2)}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Mise:</span>
                      <span className="ml-2 font-medium">{formatAriary(FIXED_STAKE_ARIARY)}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Résultat:</span>
                      <span className={`ml-2 font-bold ${betResult.is_win ? 'text-green-500' : 'text-red-500'}`}>
                        {betResult.is_win ? 'GAGNÉ' : 'PERDU'}
                      </span>
                    </div>
                  </div>
                  
                  <div className="mt-3 pt-3 border-t border-gray-600">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Gain/Perte:</span>
                      <span className={`text-xl font-bold ${betResult.profit_loss && betResult.profit_loss >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                        {betResult.profit_loss && betResult.profit_loss >= 0 ? '+' : ''}{formatAriary(FIXED_STAKE_ARIARY * (betResult.profit_loss || 0))}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Odds Info */}
              <div className="text-sm text-gray-400">
                Cotes: {selectedPrediction.match.odd_home?.toFixed(2)} (1) | {selectedPrediction.match.odd_draw?.toFixed(2)} (X) | {selectedPrediction.match.odd_away?.toFixed(2)} (2)
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Results Comparison Modal */}
      {showResultsModal && comparisonData && (
        <PredictionComparisonModal 
          data={comparisonData} 
          onClose={() => setShowResultsModal(false)} 
        />
      )}
    </div>
  )
}
