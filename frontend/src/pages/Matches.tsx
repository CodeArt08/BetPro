import { useState, useEffect } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import api from '../services/api'
import { useRefreshOnUpdate } from '../hooks/useRealTimeUpdates'

interface Match {
  id: number
  matchday: number
  line_position: number
  home_team_name: string
  away_team_name: string
  score_home: number | null
  score_away: number | null
  result: string | null
  odd_home: number | null
  odd_draw: number | null
  odd_away: number | null
  is_completed: boolean
  is_upcoming: boolean
}

export default function Matches() {
  const [matches, setMatches] = useState<Match[]>([])
  const [loading, setLoading] = useState(true)
  const [currentMatchday, setCurrentMatchday] = useState<number | null>(null)
  const [maxMatchday, setMaxMatchday] = useState<number | null>(null)

  const initMatchday = async () => {
    try {
      // Get latest matchday from active season
      const latestData = await api.getLatestMatchday()
      if (latestData.matchday > 0) {
        setMaxMatchday(latestData.matchday + 1)
        setCurrentMatchday((prev) => prev ?? latestData.matchday)
      } else {
        // No matches in active season
        setLoading(false)
      }
    } catch (error) {
      console.error('Failed to init matches:', error)
      setLoading(false)
    }
  }

  const loadMatchday = async (matchday: number) => {
    setLoading(true)
    try {
      const data = await api.getMatchday(matchday)
      setMatches(data.matches || [])
    } catch (error) {
      console.error('Failed to load matchday:', error)
    } finally {
      setLoading(false)
    }
  }

  useRefreshOnUpdate(() => {
    if (currentMatchday !== null) {
      loadMatchday(currentMatchday)
    }
    initMatchday()
  })

  useEffect(() => {
    initMatchday()
  }, [])

  useEffect(() => {
    if (currentMatchday !== null) {
      loadMatchday(currentMatchday)
    }
  }, [currentMatchday])

  if (loading && matches.length === 0) {
    return <div className="text-center py-10">Chargement des matchs...</div>
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Matches (historique illimité)</h1>
        
        {currentMatchday !== null && (
          <div className="flex items-center space-x-4">
            <button 
              disabled={currentMatchday >= (maxMatchday || 38)}
              onClick={() => setCurrentMatchday(prev => (prev || 0) + 1)}
              className="flex items-center px-3 py-2 bg-gray-700 rounded-lg hover:bg-gray-600 disabled:opacity-50"
              title="Journée plus récente"
            >
              <ChevronLeft className="w-5 h-5 mr-1" />
              Plus récent
            </button>
            <span className="font-bold text-lg px-4 py-2 bg-gray-800 rounded">
              Journée {currentMatchday}
            </span>
            <button 
              disabled={currentMatchday <= 1}
              onClick={() => setCurrentMatchday(prev => (prev || 2) - 1)}
              className="flex items-center px-3 py-2 bg-blue-600 rounded-lg hover:bg-blue-500 disabled:opacity-50"
              title="Journée plus ancienne"
            >
              Suivant (J-1)
              <ChevronRight className="w-5 h-5 ml-1" />
            </button>
          </div>
        )}
      </div>

      <div className="card">
        {matches.length === 0 && !loading ? (
          <div className="text-center py-10 text-gray-400">
            Aucun match trouvé pour la Journée {currentMatchday}.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-400 text-sm border-b border-gray-700">
                  <th className="pb-2">MD</th>
                  <th className="pb-2">Ligne</th>
                  <th className="pb-2">Match</th>
                  <th className="pb-2">Score</th>
                  <th className="pb-2">Résultat</th>
                  <th className="pb-2">Cotes (1|X|2)</th>
                  <th className="pb-2">Statut</th>
                </tr>
              </thead>
              <tbody className={loading ? 'opacity-50' : ''}>
                {matches.map((match) => (
                  <tr key={match.id} className="border-b border-gray-700 hover:bg-gray-750">
                    <td className="py-3 text-gray-400">{match.matchday}</td>
                    <td className="py-3 text-gray-400">{match.line_position}</td>
                    <td className="py-3">
                      <span className="font-medium">{match.home_team_name}</span>
                      <span className="text-gray-400 mx-2">vs</span>
                      <span className="font-medium">{match.away_team_name}</span>
                    </td>
                    <td className="py-3 font-bold">
                      {match.is_completed ? `${match.score_home} - ${match.score_away}` : '-'}
                    </td>
                    <td className="py-3">
                      {match.result && (
                        <span className={`px-2 py-1 rounded text-xs font-bold ${
                          match.result === 'V' ? 'bg-green-600' :
                          match.result === 'N' ? 'bg-yellow-600' : 'bg-red-600'
                        }`}>
                          {match.result === 'V' ? '1' : match.result === 'N' ? 'X' : '2'}
                        </span>
                      )}
                    </td>
                    <td className="py-3 text-sm text-gray-400">
                      {match.odd_home ? `${match.odd_home.toFixed(2)} | ${match.odd_draw?.toFixed(2)} | ${match.odd_away?.toFixed(2)}` : '-'}
                    </td>
                    <td className="py-3">
                      <span className={`px-2 py-1 rounded text-xs ${
                        match.is_completed ? 'bg-green-900 text-green-300' : 'bg-blue-900 text-blue-300'
                      }`}>
                        {match.is_completed ? 'Terminé' : 'À venir'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
