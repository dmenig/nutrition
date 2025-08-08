package com.nutrition.app.ui.foodentry

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.nutrition.app.data.NutritionRepository
import com.nutrition.app.data.local.entities.FoodLog
import com.nutrition.app.data.repository.NutritionRepository as LocalNutritionRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.launch
import java.time.LocalDateTime
import java.time.ZoneId
import javax.inject.Inject

@HiltViewModel
class FoodEntryViewModel @Inject constructor(
    private val nutritionRepository: NutritionRepository,
    private val localRepository: LocalNutritionRepository,
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
                // Heuristic aligned with backend/demo population:
                // If quantity <= 10, interpret as number of 100g servings; else treat as grams.
                val quantityInGrams = if (parsedQuantity <= 10f) parsedQuantity * 100f else parsedQuantity
                // Nutriments from backend are per 100g. Quantity is in grams.
                // Scale per-100g values by (grams / 100) to get actual nutrient amounts.
                val scale = quantityInGrams / 100f
                val calories = food.nutriments.calories * scale
                val protein = food.nutriments.protein * scale
                val carbs = food.nutriments.carbohydrates * scale
                val fat = food.nutriments.fat * scale

                // Persist locally so Daily Log updates immediately
                val localEpochMillis = loggedAt.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli()
                localRepository.insertFoodLog(
                    FoodLog(
                        foodName = foodName,
                        calories = calories.toDouble(),
                        protein = protein.toDouble(),
                        carbs = carbs.toDouble(),
                        fat = fat.toDouble(),
                        date = localEpochMillis
                    )
                )

                // Attempt remote creation
                // Send normalized quantity and unit to backend
                val result = nutritionRepository.insertFoodLog(
                    foodName, quantityInGrams, "g", loggedAt, calories, protein, carbs, fat
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