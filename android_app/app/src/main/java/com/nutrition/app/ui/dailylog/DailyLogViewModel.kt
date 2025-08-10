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
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.util.Date
import javax.inject.Inject
import java.time.ZoneId
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZoneOffset

@OptIn(ExperimentalCoroutinesApi::class)
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
        // Daily summary flow tied to selected date
        viewModelScope.launch {
            _selectedDate
                .flatMapLatest { date ->
                    val start = date.time.atStartOfDay()
                    val end = date.time.atEndOfDay()
                    repository.getDailySummary(start, end)
                }
                .collect { summary -> _dailySummary.value = summary }
        }

        // Food logs flow tied to selected date; when empty, fetch from remote in bulk
        viewModelScope.launch {
            _selectedDate
                .flatMapLatest { date ->
                    val start = date.time.atStartOfDay()
                    val end = date.time.atEndOfDay()
                    repository.getLogsForDay(start, end)
                }
                .collect { foodLogs ->
                    _foodLogs.value = foodLogs
                    if (foodLogs.isEmpty()) {
                        try {
                            val date = selectedDate.value
                            val localDate = date.toInstant().atZone(ZoneId.systemDefault()).toLocalDate()
                            val remoteLogs = remoteRepository.fetchFoodLogsForDate(localDate).getOrNull()
                            if (!remoteLogs.isNullOrEmpty()) {
                                val toInsert = remoteLogs.map { rl: FoodLogResponse ->
                                    val epochMillis = toEpochMillisUtc(rl.loggedAt)
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
                                }
                                repository.insertFoodLogs(toInsert)
                            }
                        } catch (_: Exception) { }
                    }
                }
        }

        // Sport logs flow tied to selected date; refresh from remote in bulk when needed
        viewModelScope.launch {
            _selectedDate
                .flatMapLatest { date ->
                    val start = date.time.atStartOfDay()
                    val end = date.time.atEndOfDay()
                    repository.getActivitiesForDay(start, end)
                }
                .collect { sportActivities ->
                    _sportLogs.value = sportActivities

                    val shouldRefreshFromRemote = sportActivities.isEmpty() ||
                        sportActivities.any { sa ->
                            val needsDistanceOrWeight = sa.activityName in listOf("Walking", "Running", "Cycling")
                            needsDistanceOrWeight && (sa.distanceM == null || sa.carriedWeightKg == null)
                        }

                    if (shouldRefreshFromRemote) {
                        try {
                            val date = selectedDate.value
                            val start = date.time.atStartOfDay()
                            val end = date.time.atEndOfDay()
                            val localDate = date.toInstant().atZone(ZoneId.systemDefault()).toLocalDate()
                            val remoteSports = remoteRepository.fetchSportActivitiesForDate(localDate).getOrNull()
                            if (!remoteSports.isNullOrEmpty()) {
                                val toInsert = remoteSports.map { rs: SportActivityResponse ->
                                    val epochMillis = toEpochMillisUtc(rs.loggedAt)
                                    com.nutrition.app.data.local.entities.SportActivity(
                                        activityName = rs.activityName,
                                        durationMinutes = rs.durationMinutes,
                                        carriedWeightKg = rs.carriedWeightKg?.toDouble(),
                                        distanceM = rs.distanceM?.toDouble(),
                                        caloriesBurned = rs.caloriesExpended.toDouble(),
                                        date = epochMillis,
                                        synced = true
                                    )
                                }
                                repository.deleteActivitiesForDay(start, end)
                                repository.insertSportActivities(toInsert)
                            }
                        } catch (_: Exception) { }
                    }
                }
        }
    }

    fun selectDate(date: Date) {
        _selectedDate.value = date
    }

    private fun toEpochMillis(dt: LocalDateTime): Long =
        dt.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli()

    private fun toEpochMillisUtc(dt: LocalDateTime): Long =
        dt.atZone(ZoneOffset.UTC).toInstant().toEpochMilli()
}