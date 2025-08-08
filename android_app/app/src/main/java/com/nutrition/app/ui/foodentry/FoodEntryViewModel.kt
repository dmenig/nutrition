package com.nutrition.app.ui.foodentry

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.nutrition.app.data.NutritionRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.launch
import java.time.LocalDateTime
import javax.inject.Inject

@HiltViewModel
class FoodEntryViewModel @Inject constructor(
    private val nutritionRepository: NutritionRepository,
    private val savedStateHandle: SavedStateHandle
) : ViewModel() {

    private val _uiEvent = Channel<FoodEntryUiEvent>()
    val uiEvent = _uiEvent.receiveAsFlow()

    fun saveFoodLogEntry(
        foodName: String,
        quantity: String,
        loggedAt: LocalDateTime
    ) {
        viewModelScope.launch {
            val food = nutritionRepository.searchFoods(foodName).getOrNull()?.firstOrNull()
            if (food != null) {
                val parsedQuantity = quantity.toFloatOrNull() ?: 0f
                val calories = food.nutriments.calories * parsedQuantity
                val protein = food.nutriments.protein * parsedQuantity
                val carbs = food.nutriments.carbohydrates * parsedQuantity
                val fat = food.nutriments.fat * parsedQuantity

                val result = nutritionRepository.insertFoodLog(
                    foodName, parsedQuantity, "g", loggedAt, calories, protein, carbs, fat
                )
                if (result.isSuccess) {
                    _uiEvent.send(FoodEntryUiEvent.FoodLogSaveSuccess)
                } else {
                    _uiEvent.send(FoodEntryUiEvent.ShowError("Failed to save food log: ${result.exceptionOrNull()?.message}"))
                }
            } else {
                _uiEvent.send(FoodEntryUiEvent.ShowError("Food '$foodName' not found. Please enter a valid food name."))
            }
        }
    }
}

sealed class FoodEntryUiEvent {
    object FoodLogSaveSuccess : FoodEntryUiEvent()
    data class ShowError(val message: String) : FoodEntryUiEvent()
}