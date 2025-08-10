package com.nutrition.app.ui.plots

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.github.mikephil.charting.data.Entry
import com.nutrition.app.data.PlotsRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject
import android.util.Log
import com.nutrition.app.data.DateRange as DataDateRange


@HiltViewModel
class PlotsViewModel @Inject constructor(
    private val plotsRepository: PlotsRepository
) : ViewModel() {

    private val _weightData = MutableStateFlow<List<Entry>>(emptyList())
    val weightData: StateFlow<List<Entry>> = _weightData

    private val _metabolismData = MutableStateFlow<List<Entry>>(emptyList())
    val metabolismData: StateFlow<List<Entry>> = _metabolismData

    private val _energyBalanceData = MutableStateFlow<List<Entry>>(emptyList())
    val energyBalanceData: StateFlow<List<Entry>> = _energyBalanceData

    private val _selectedDateRange = MutableStateFlow(DateRange.LAST_7_DAYS)
    val selectedDateRange: StateFlow<DateRange> = _selectedDateRange

    init {
        fetchWeightData(DateRange.LAST_7_DAYS)
        fetchMetabolismData(DateRange.LAST_7_DAYS)
        fetchEnergyBalanceData(DateRange.LAST_7_DAYS)
    }

    fun setDateRange(dateRange: DateRange) {
        _selectedDateRange.value = dateRange
        fetchWeightData(dateRange)
        fetchMetabolismData(dateRange)
        fetchEnergyBalanceData(dateRange)
    }

    private fun toDataDateRange(uiDateRange: DateRange): DataDateRange {
        return when (uiDateRange) {
            DateRange.LAST_7_DAYS -> DataDateRange.WEEK
            DateRange.LAST_30_DAYS -> DataDateRange.MONTH
            DateRange.ALL_TIME -> DataDateRange.YEAR
        }
    }

    private fun fetchWeightData(dateRange: DateRange) {
        viewModelScope.launch {
            try {
                _weightData.value = plotsRepository.getWeightData(toDataDateRange(dateRange))
            } catch (e: Exception) {
                Log.e("PlotsViewModel", "Failed to fetch weight data", e)
                _weightData.value = emptyList()
            }
        }
    }

    private fun fetchMetabolismData(dateRange: DateRange) {
        viewModelScope.launch {
            try {
                _metabolismData.value = plotsRepository.getMetabolismData(toDataDateRange(dateRange))
            } catch (e: Exception) {
                Log.e("PlotsViewModel", "Failed to fetch metabolism data", e)
                _metabolismData.value = emptyList()
            }
        }
    }

    private fun fetchEnergyBalanceData(dateRange: DateRange) {
        viewModelScope.launch {
            try {
                _energyBalanceData.value = plotsRepository.getEnergyBalanceData(toDataDateRange(dateRange))
            } catch (e: Exception) {
                Log.e("PlotsViewModel", "Failed to fetch energy balance data", e)
                _energyBalanceData.value = emptyList()
            }
        }
    }
}