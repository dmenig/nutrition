package com.nutrition.app.ui.dailylog

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.nutrition.app.data.local.dao.DailyNutritionSummary
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

@HiltViewModel
class DailyLogViewModel @Inject constructor(
    private val repository: NutritionRepository
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
            }
        }

        viewModelScope.launch {
            repository.getActivitiesForDay(startOfDay, endOfDay).collect { sportActivities ->
                _sportLogs.value = sportActivities
            }
        }
    }
}