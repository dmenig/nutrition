package com.nutrition.app.ui.sportentry

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.nutrition.app.data.NutritionRepository
import com.nutrition.app.data.local.entities.SportActivity
import com.nutrition.app.data.repository.NutritionRepository as LocalNutritionRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.launch
import java.time.LocalDateTime
import java.time.ZoneId
import javax.inject.Inject

@HiltViewModel
class SportEntryViewModel @Inject constructor(
    private val nutritionRepository: NutritionRepository,
    private val localRepository: LocalNutritionRepository,
    private val savedStateHandle: SavedStateHandle
) : ViewModel() {

    private val _uiEvent = Channel<SportEntryUiEvent>()
    val uiEvent = _uiEvent.receiveAsFlow()

    private val _sportNames = MutableStateFlow<List<String>>(emptyList())
    val sportNames = _sportNames.asStateFlow()

    init {
        fetchSportNames()
    }

    private fun fetchSportNames() {
        viewModelScope.launch {
            val result = nutritionRepository.getSportNames()
            val names = result.getOrElse { emptyList() }
            _sportNames.value = if (names.isNotEmpty()) names else DEFAULT_SPORT_NAMES
        }
    }

    fun saveSportActivityEntry(
        activityName: String,
        loggedAt: LocalDateTime,
        durationMinutes: Int,
        carriedWeightKg: Float?,
        distanceM: Float?
    ) {
        viewModelScope.launch {
            // Persist locally so Daily Log updates immediately
            val localEpochMillis = loggedAt.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli()
            localRepository.insertSportActivity(
                SportActivity(
                    activityName = activityName,
                    durationMinutes = durationMinutes,
                    caloriesBurned = 0.0,
                    date = localEpochMillis
                )
            )

            // Attempt remote creation
            nutritionRepository.insertSportActivity(
                activityName, loggedAt, durationMinutes, carriedWeightKg, distanceM
            )
            _uiEvent.send(SportEntryUiEvent.SportActivitySaveSuccess)
        }
    }
}

sealed class SportEntryUiEvent {
    object SportActivitySaveSuccess : SportEntryUiEvent()
    data class ShowError(val message: String) : SportEntryUiEvent()
}

private val DEFAULT_SPORT_NAMES = listOf(
    "Walking",
    "Running",
    "Cycling",
    "Swimming",
    "Rowing",
    "Elliptical",
    "Hiking",
    "Yoga",
    "Strength Training",
    "Pilates"
)