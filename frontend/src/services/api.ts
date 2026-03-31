import axios from 'axios'

const API_BASE = '/api'

// Base axios instance (for generic calls in components)
const axiosInstance = axios.create({ baseURL: '' })

export const api = {
  // Generic axios passthrough for components
  get: axiosInstance.get.bind(axiosInstance),
  post: axiosInstance.post.bind(axiosInstance),


  // Dashboard
  async getOverview() {
    const { data } = await axios.get(`${API_BASE}/dashboard/overview`)
    return data
  },

  async getStandings() {
    const { data } = await axios.get(`${API_BASE}/dashboard/standings`)
    return data
  },

  async getStatistics() {
    const { data } = await axios.get(`${API_BASE}/dashboard/statistics`)
    return data
  },

  // Matches
  async getMatches(params?: { season_id?: number; matchday?: number; is_completed?: boolean }) {
    const { data } = await axios.get(`${API_BASE}/matches/`, { params })
    return data
  },

  async getUpcomingMatches() {
    const { data } = await axios.get(`${API_BASE}/matches/upcoming`)
    return data
  },

  async getRecentMatches() {
    const { data } = await axios.get(`${API_BASE}/matches/recent`)
    return data
  },

  async getMatchday(matchday: number) {
    const { data } = await axios.get(`${API_BASE}/matches/matchday/${matchday}`)
    return data
  },

  async getLatestMatchday() {
    const { data } = await axios.get(`${API_BASE}/matches/latest-matchday`)
    return data
  },

  // Predictions
  async getPredictions(params?: { season_id?: number; is_verified?: boolean }) {
    const { data } = await axios.get(`${API_BASE}/predictions/`, { params })
    return data
  },

  async getUpcomingPredictions() {
    const { data } = await axios.get(`${API_BASE}/predictions/upcoming`)
    return data
  },

  async getPredictionAccuracy() {
    const { data } = await axios.get(`${API_BASE}/predictions/accuracy/stats`)
    return data
  },

  async getMatchdayComparison(matchday: number) {
    const { data } = await axios.get(`${API_BASE}/predictions/comparison/${matchday}`)
    return data
  },

  async getAllComparisons() {
    const { data } = await axios.get(`${API_BASE}/predictions/comparison`)
    return data
  },

  async getVerifiedMatchdays() {
    const { data } = await axios.get(`${API_BASE}/predictions/verified-matchdays`)
    return data
  },

  // Historical season stats
  async getSeasonBettingHistory() {
    const { data } = await axios.get(`${API_BASE}/predictions/season-history`)
    return data
  },

  // Dynamic Selection - NEW
  async getDynamicSelection(matchday: number, bankroll?: number) {
    const { data } = await axios.get(`${API_BASE}/predictions/dynamic-selection/${matchday}`, {
      params: { bankroll: bankroll || 10000 }
    })
    return data
  },

  // Selected predictions only - NEW
  async getSelectedPredictions(limit?: number) {
    const { data } = await axios.get(`${API_BASE}/predictions/selected`, {
      params: { limit: limit || 20 }
    })
    return data
  },

  // Recalculate values and run selection - NEW
  async recalculateSelection() {
    const { data } = await axios.post(`${API_BASE}/predictions/recalculate-selection`)
    return data
  },

  // Betting - kept for prediction details
  async getBetForMatch(matchId: number) {
    const { data } = await axios.get(`${API_BASE}/betting/match/${matchId}`)
    return data
  },

  // Teams
  async getTeams() {
    const { data } = await axios.get(`${API_BASE}/teams/`)
    return data
  },

  async getEloRankings() {
    const { data } = await axios.get(`${API_BASE}/teams/elo-rankings`)
    return data
  },

  async getTeamMatches(teamId: number) {
    const { data } = await axios.get(`${API_BASE}/teams/${teamId}/matches`)
    return data
  },

  async compareTeams(team1: string, team2: string) {
    const { data } = await axios.get(`${API_BASE}/teams/comparison`, { params: { team1, team2 } })
    return data
  },

  // Learning
  async getLearningStatus() {
    const { data } = await axios.get(`${API_BASE}/dashboard/learning-status`)
    return data
  },

  // Scrape
  async scrapeResults() {
    const { data } = await axios.post(`${API_BASE}/scrape/results`)
    return data
  },

  async scrapeMatches() {
    const { data } = await axios.post(`${API_BASE}/scrape/matches`)
    return data
  },

  async createNewSeason() {
    const { data } = await axios.post(`${API_BASE}/scrape/new-season`)
    return data
  },
}

export default api
