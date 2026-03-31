import { X, CheckCircle, XCircle, DollarSign, ChevronDown, ChevronUp } from 'lucide-react'
import { useState } from 'react'

interface PredictionResult {
  match: string
  matchday: number
  predicted: string
  actual: string
  actual_result: string | null
  is_correct: boolean
  stake: number
  profit_loss: number
  score: string | null
  odds?: number
  predicted_result?: string
}

interface MatchdayResult {
  matchday: number
  predictions: PredictionResult[]
  stake: number
  profit: number
  correct: number
}

interface DNFilterResult {
  predictions: PredictionResult[]
  total_stake: number
  total_profit: number
  correct_count: number
  total_predictions: number
  accuracy: number
  description: string
}

interface ComparisonData {
  predictions: PredictionResult[]
  by_matchday?: MatchdayResult[]
  total_stake: number
  total_profit: number
  correct_count: number
  total_predictions: number
  accuracy: number
  timestamp: string
  dn_filter?: DNFilterResult
}

interface Props {
  data: ComparisonData | null
  onClose: () => void
}

export default function PredictionComparisonModal({ data, onClose }: Props) {
  const [expandedMatchdays, setExpandedMatchdays] = useState<Set<number>>(new Set())
  
  if (!data) return null

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('mg-MG', {
      style: 'decimal',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount) + ' Ar'
  }
  
  const toggleMatchday = (md: number) => {
    const newExpanded = new Set(expandedMatchdays)
    if (newExpanded.has(md)) {
      newExpanded.delete(md)
    } else {
      newExpanded.add(md)
    }
    setExpandedMatchdays(newExpanded)
  }
  
  // Use grouped data if available, otherwise fall back to flat list
  const matchdayGroups = data.by_matchday || (data.predictions.length > 0 ? [{
    matchday: data.predictions[0].matchday,
    predictions: data.predictions,
    stake: data.total_stake,
    profit: data.total_profit,
    correct: data.correct_count
  }] : [])

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div 
        className="bg-gray-800 rounded-xl max-w-2xl w-full max-h-[90vh] overflow-hidden shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-4 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <DollarSign className="w-6 h-6 text-white" />
            <h2 className="text-xl font-bold text-white">Résultat des Prédictions</h2>
          </div>
          <button 
            onClick={onClose}
            className="text-white/80 hover:text-white transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Summary */}
        <div className="p-4 bg-gray-750 border-b border-gray-700">
          <div className="grid grid-cols-4 gap-4 text-center">
            <div className="bg-gray-700 rounded-lg p-3">
              <div className="text-gray-400 text-xs mb-1">Total Misé</div>
              <div className="text-lg font-bold text-white">{formatCurrency(data.total_stake)}</div>
            </div>
            <div className="bg-gray-700 rounded-lg p-3">
              <div className="text-gray-400 text-xs mb-1">Gain/Perte</div>
              <div className={`text-lg font-bold ${data.total_profit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {data.total_profit >= 0 ? '+' : ''}{formatCurrency(data.total_profit)}
              </div>
            </div>
            <div className="bg-gray-700 rounded-lg p-3">
              <div className="text-gray-400 text-xs mb-1">Précision</div>
              <div className="text-lg font-bold text-blue-400">
                {(data.accuracy * 100).toFixed(0)}%
              </div>
            </div>
            <div className="bg-gray-700 rounded-lg p-3">
              <div className="text-gray-400 text-xs mb-1">Correctes</div>
              <div className="text-lg font-bold text-white">
                {data.correct_count}/{data.total_predictions}
              </div>
            </div>
          </div>
          
          {/* DN Filter Summary */}
          {data.dn_filter && data.dn_filter.total_predictions > 0 && (
            <div className="mt-4 bg-gradient-to-r from-purple-900/50 to-blue-900/50 rounded-lg p-4 border border-purple-500/30">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <span className="text-purple-400 font-bold text-sm">📊 Filtre D/N</span>
                  <span className="text-xs text-gray-400">(D/N avec cote &gt; 2.00)</span>
                </div>
                <span className="text-xs text-purple-300 bg-purple-800/50 px-2 py-1 rounded">
                  {data.dn_filter.total_predictions} prédictions
                </span>
              </div>
              <div className="grid grid-cols-4 gap-3 text-center">
                <div>
                  <div className="text-gray-400 text-xs">Misé</div>
                  <div className="text-sm font-bold text-white">{formatCurrency(data.dn_filter.total_stake)}</div>
                </div>
                <div>
                  <div className="text-gray-400 text-xs">Gain/Perte</div>
                  <div className={`text-sm font-bold ${data.dn_filter.total_profit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {data.dn_filter.total_profit >= 0 ? '+' : ''}{formatCurrency(data.dn_filter.total_profit)}
                  </div>
                </div>
                <div>
                  <div className="text-gray-400 text-xs">Précision</div>
                  <div className="text-sm font-bold text-purple-400">
                    {(data.dn_filter.accuracy * 100).toFixed(0)}%
                  </div>
                </div>
                <div>
                  <div className="text-gray-400 text-xs">Correctes</div>
                  <div className="text-sm font-bold text-white">
                    {data.dn_filter.correct_count}/{data.dn_filter.total_predictions}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Predictions List - Grouped by Matchday */}
        <div className="overflow-y-auto max-h-[50vh] p-4">
          {matchdayGroups.length === 0 ? (
            <p className="text-gray-400 text-center">Aucune prédiction à comparer</p>
          ) : (
            <div className="space-y-3">
              {matchdayGroups.map((group) => (
                <div key={group.matchday} className="bg-gray-700 rounded-lg overflow-hidden">
                  {/* Matchday Header */}
                  <button 
                    className="w-full p-3 flex justify-between items-center hover:bg-gray-600 transition-colors"
                    onClick={() => toggleMatchday(group.matchday)}
                  >
                    <div className="flex items-center space-x-3">
                      <span className="text-lg font-bold text-blue-400">J{group.matchday}</span>
                      <span className="text-sm text-gray-400">
                        {group.correct}/{group.predictions.length} correct
                      </span>
                      <span className={`text-sm font-bold ${group.profit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {group.profit >= 0 ? '+' : ''}{formatCurrency(group.profit)}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs text-gray-400">
                        {(group.correct / group.predictions.length * 100).toFixed(0)}%
                      </span>
                      {expandedMatchdays.has(group.matchday) ? (
                        <ChevronUp className="w-5 h-5 text-gray-400" />
                      ) : (
                        <ChevronDown className="w-5 h-5 text-gray-400" />
                      )}
                    </div>
                  </button>
                  
                  {/* Expanded Predictions */}
                  {expandedMatchdays.has(group.matchday) && (
                    <div className="border-t border-gray-600">
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
                          {group.predictions.map((pred, idx) => (
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
                                    {pred.profit_loss >= 0 ? '+' : ''}{formatCurrency(pred.profit_loss)}
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
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 bg-gray-750 border-t border-gray-700 flex justify-between items-center">
          <div className="text-sm text-gray-400">
            Mise fixe: <span className="font-bold text-white">1 000 Ar</span> par prédiction
          </div>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-white font-medium transition-colors"
          >
            Fermer
          </button>
        </div>
      </div>
    </div>
  )
}
