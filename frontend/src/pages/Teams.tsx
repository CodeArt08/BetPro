import { useState, useEffect } from 'react'
import { TrendingUp, Shield, Sword } from 'lucide-react'
import api from '../services/api'
import { useRefreshOnUpdate } from '../hooks/useRealTimeUpdates'

interface Team {
  id: number
  name: string
  elo_rating: number
  elo_home: number
  elo_away: number
  bayesian_rating: number
  attack_strength: number
  defense_strength: number
  matches_played: number
  wins: number
  draws: number
  losses: number
  goals_scored: number
  goals_conceded: number
  current_form: string
  winning_streak: number
  losing_streak: number
}

interface Standing {
  position: number
  team_id: number
  team_name: string
  played: number
  won: number
  drawn: number
  lost: number
  goals_for: number
  goals_against: number
  goal_difference: number
  points: number
}

export default function Teams() {
  const [teams, setTeams] = useState<Team[]>([])
  const [standings, setStandings] = useState<Standing[]>([])
  const [eloRankings, setEloRankings] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [view, setView] = useState<'standings' | 'elo' | 'strength'>('standings')
  const loadTeams = async () => {
    try {
      const [teamsData, standingsData, eloData] = await Promise.all([
        api.getTeams(),
        api.getStandings(),
        api.getEloRankings()
      ])
      setTeams(teamsData)
      setStandings(standingsData.standings || [])
      setEloRankings(eloData.rankings || [])
    } catch (error) {
      console.error('Failed to load teams:', error)
    } finally {
      setLoading(false)
    }
  }

  useRefreshOnUpdate(loadTeams)

  useEffect(() => {
    loadTeams()
  }, [])

  if (loading) {
    return <div className="text-center py-10">Loading teams...</div>
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Teams</h1>
        <div className="flex space-x-2">
          {[
            { key: 'standings', label: 'Standings' },
            { key: 'elo', label: 'ELO Ratings' },
            { key: 'strength', label: 'Strength' }
          ].map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setView(key as any)}
              className={`px-4 py-2 rounded-lg ${
                view === key ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Standings View */}
      {view === 'standings' && (
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">League Standings</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-400 text-sm border-b border-gray-700">
                  <th className="pb-2">#</th>
                  <th className="pb-2">Team</th>
                  <th className="pb-2">P</th>
                  <th className="pb-2">W</th>
                  <th className="pb-2">D</th>
                  <th className="pb-2">L</th>
                  <th className="pb-2">GF</th>
                  <th className="pb-2">GA</th>
                  <th className="pb-2">GD</th>
                  <th className="pb-2">Pts</th>
                </tr>
              </thead>
              <tbody>
                {standings.map((team, idx) => (
                  <tr key={team.team_id || idx} className="border-b border-gray-700 hover:bg-gray-750">
                    <td className="py-3 font-bold">{team.position}</td>
                    <td className="py-3 font-medium">{team.team_name}</td>
                    <td className="py-3 text-gray-400">{team.played}</td>
                    <td className="py-3 text-green-400">{team.won}</td>
                    <td className="py-3 text-yellow-400">{team.drawn}</td>
                    <td className="py-3 text-red-400">{team.lost}</td>
                    <td className="py-3">{team.goals_for}</td>
                    <td className="py-3">{team.goals_against}</td>
                    <td className="py-3 font-bold">{team.goal_difference > 0 ? '+' : ''}{team.goal_difference}</td>
                    <td className="py-3 font-bold text-blue-400">{team.points}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ELO Ratings View */}
      {view === 'elo' && (
        <div className="card">
          <h2 className="text-lg font-semibold mb-4 flex items-center">
            <TrendingUp className="w-5 h-5 mr-2" />
            ELO Ratings
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-400 text-sm border-b border-gray-700">
                  <th className="pb-2">#</th>
                  <th className="pb-2">Team</th>
                  <th className="pb-2">Overall</th>
                  <th className="pb-2">Home</th>
                  <th className="pb-2">Away</th>
                  <th className="pb-2">Matches</th>
                  <th className="pb-2">Win Rate</th>
                </tr>
              </thead>
              <tbody>
                {eloRankings.map((team) => (
                  <tr key={team.position} className="border-b border-gray-700 hover:bg-gray-750">
                    <td className="py-3 font-bold">{team.position}</td>
                    <td className="py-3 font-medium">{team.name}</td>
                    <td className="py-3 text-blue-400 font-bold">{team.elo_rating.toFixed(1)}</td>
                    <td className="py-3 text-green-400">{team.elo_home.toFixed(1)}</td>
                    <td className="py-3 text-red-400">{team.elo_away.toFixed(1)}</td>
                    <td className="py-3 text-gray-400">{team.matches_played}</td>
                    <td className="py-3">{(team.win_rate * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Strength View */}
      {view === 'strength' && (
        <div className="card">
          <h2 className="text-lg font-semibold mb-4 flex items-center">
            <Sword className="w-5 h-5 mr-2" />
            Team Strength
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-400 text-sm border-b border-gray-700">
                  <th className="pb-2">Team</th>
                  <th className="pb-2">Attack</th>
                  <th className="pb-2">Defense</th>
                  <th className="pb-2">Form</th>
                  <th className="pb-2">W-Streak</th>
                  <th className="pb-2">L-Streak</th>
                </tr>
              </thead>
              <tbody>
                {teams.map((team) => (
                  <tr key={team.id} className="border-b border-gray-700 hover:bg-gray-750">
                    <td className="py-3 font-medium">{team.name}</td>
                    <td className="py-3">
                      <div className="flex items-center">
                        <Sword className="w-4 h-4 text-red-400 mr-1" />
                        <span className={team.attack_strength >= 1 ? 'text-green-400' : 'text-red-400'}>
                          {team.attack_strength.toFixed(2)}
                        </span>
                      </div>
                    </td>
                    <td className="py-3">
                      <div className="flex items-center">
                        <Shield className="w-4 h-4 text-blue-400 mr-1" />
                        <span className={team.defense_strength <= 1 ? 'text-green-400' : 'text-red-400'}>
                          {team.defense_strength.toFixed(2)}
                        </span>
                      </div>
                    </td>
                    <td className="py-3">
                      <FormDisplay form={team.current_form} />
                    </td>
                    <td className="py-3 text-green-400">{team.winning_streak}</td>
                    <td className="py-3 text-red-400">{team.losing_streak}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

function FormDisplay({ form }: { form: string }) {
  if (!form) return <span className="text-gray-400">-</span>
  
  const results = form.slice(-5).split('')
  
  return (
    <div className="flex space-x-1">
      {results.map((r, idx) => (
        <span
          key={idx}
          className={`w-5 h-5 flex items-center justify-center rounded text-xs font-bold ${
            r === 'V' ? 'bg-green-600' : r === 'N' ? 'bg-yellow-600' : 'bg-red-600'
          }`}
        >
          {r}
        </span>
      ))}
    </div>
  )
}
