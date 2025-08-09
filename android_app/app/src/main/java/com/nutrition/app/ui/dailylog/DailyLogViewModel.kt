package com.nutrition.app.ui.dailylog

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.nutrition.app.data.local.dao.DailyNutritionSummary
import com.nutrition.app.data.remote.model.FoodLogResponse
import com.nutrition.app.data.remote.model.SportActivityResponse
import com.nutrition.app.data.local.entities.FoodLog
import com.nutrition.app.data.local.entities.SportActivity
import com.nutrition.app.data.repository.NutritionRepository
import com.nutrition.app.util.atEndOfDay
import com.nutrition.app.util.atStartOfDay
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import java.util.Date
import javax.inject.Inject
import java.time.ZoneId
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZoneOffset

@HiltViewModel
class DailyLogViewModel @Inject constructor(
    private val repository: NutritionRepository,
    private val remoteRepository: com.nutrition.app.data.NutritionRepository
) : ViewModel() {

    private val _selectedDate = MutableStateFlow(Date())
    val selectedDate: StateFlow<Date> = _selectedDate

    private val _dailySummary = MutableStateFlow<DailyNutritionSummary?>(null)
    val dailySummary: StateFlow<DailyNutritionSummary?> = _dailySummary

    private val _foodLogs = MutableStateFlow<List<FoodLog>>(emptyList())
    val foodLogs: StateFlow<List<FoodLog>> = _foodLogs

    private val _sportLogs = MutableStateFlow<List<SportActivity>>(emptyList())
    val sportLogs: StateFlow<List<SportActivity>> = _sportLogs

    init {
        loadDailyData(selectedDate.value)
    }

    fun selectDate(date: Date) {
        _selectedDate.value = date
        loadDailyData(date)
    }

    private fun loadDailyData(date: Date) {
        val startOfDay = date.time.atStartOfDay()
        val endOfDay = date.time.atEndOfDay()

        viewModelScope.launch {
            repository.getDailySummary(startOfDay, endOfDay).collect { summary ->
                _dailySummary.value = summary
            }
        }

        viewModelScope.launch {
            repository.getLogsForDay(startOfDay, endOfDay).collect { foodLogs ->
                _foodLogs.value = foodLogs
                if (foodLogs.isEmpty()) {
                    // Fallback: fetch sample/public logs for demo if local is empty
                    try {
                        val localDate = date.toInstant().atZone(ZoneId.systemDefault()).toLocalDate()
                        val remoteLogs = remoteRepository.fetchFoodLogsForDate(localDate).getOrNull()
                        if (!remoteLogs.isNullOrEmpty()) {
                            remoteLogs.forEach { rl: FoodLogResponse ->
                                // Remote timestamps are UTC; convert accordingly to avoid TZ shifts
                                val epochMillis = toEpochMillisUtc(rl.loggedAt)
                                // Trust server values as-is; no rescaling heuristics.
                                repository.insertFoodLog(
                                    FoodLog(
                                        foodName = rl.foodName,
                                        quantity = rl.quantity.toDouble(),
                                        unit = rl.unit,
                                        calories = rl.calories.toDouble(),
                                        protein = rl.protein.toDouble(),
                                        carbs = rl.carbs.toDouble(),
                                        fat = rl.fat.toDouble(),
                                        date = epochMillis,
                                        synced = true
                                    )
                                )
                            }
                        }
                    } catch (_: Exception) { /* ignore demo fetch errors */ }
                }
            }
        }

        viewModelScope.launch {
            repository.getActivitiesForDay(startOfDay, endOfDay).collect { sportActivities ->
                _sportLogs.value = sportActivities
                if (sportActivities.isEmpty()) {
                    // Fallback: fetch remote sports for the date and insert locally
                    try {
                        val localDate = date.toInstant().atZone(ZoneId.systemDefault()).toLocalDate()
                        val remoteSports = remoteRepository.fetchSportActivitiesForDate(localDate).getOrNull()
                        if (!remoteSports.isNullOrEmpty()) {
                            remoteSports.forEach { rs: SportActivityResponse ->
                                // Remote timestamps are UTC; convert accordingly to avoid TZ shifts
                                val epochMillis = toEpochMillisUtc(rs.loggedAt)
                                repository.insertSportActivity(
                                    com.nutrition.app.data.local.entities.SportActivity(
                                        activityName = rs.activityName,
                                        durationMinutes = rs.durationMinutes,
                                        caloriesBurned = rs.caloriesExpended.toDouble(),
                                        date = epochMillis,
                                        synced = true
                                    )
                                )
                            }
                        }
                    } catch (_: Exception) { /* ignore demo fetch errors */ }
                }
            }
        }

        // No-op: remote sample pull happens on-demand in food logs collector when empty
    }

    private fun toEpochMillis(dt: LocalDateTime): Long =
        dt.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli()

    private fun toEpochMillisUtc(dt: LocalDateTime): Long =
        dt.atZone(ZoneOffset.UTC).toInstant().toEpochMilli()
}