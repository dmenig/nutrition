package com.nutrition.app.ui.dailylog

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.nutrition.app.data.local.dao.DailyNutritionSummary
import com.nutrition.app.data.remote.model.FoodLogResponse
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
                                val epochMillis = toEpochMillis(rl.loggedAt)
                                // Fix-up legacy demo data where quantity was interpreted as grams instead of 100g servings
                                val needsRescale = rl.unit.lowercase() == "g" && rl.quantity <= 10f
                                val factor = if (needsRescale) 100.0 else 1.0
                                repository.insertFoodLog(
                                    FoodLog(
                                        foodName = rl.foodName,
                                        calories = (rl.calories.toDouble() * factor),
                                        protein = (rl.protein.toDouble() * factor),
                                        carbs = (rl.carbs.toDouble() * factor),
                                        fat = (rl.fat.toDouble() * factor),
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
            }
        }

        // No-op: remote sample pull happens on-demand in food logs collector when empty
    }

    private fun toEpochMillis(dt: LocalDateTime): Long =
        dt.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli()
}