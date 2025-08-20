package com.nutrition.app.ui.customfood

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.nutrition.app.data.NutritionRepository
import com.nutrition.app.data.remote.model.BackendFood
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class CustomFoodViewModel @Inject constructor(
    private val repository: NutritionRepository
) : ViewModel() {

    private val _foods = MutableStateFlow<List<BackendFood>>(emptyList())
    val foods: StateFlow<List<BackendFood>> = _foods.asStateFlow()

    fun refresh(query: String) {
        viewModelScope.launch {
            val result = repository.searchFoods(query)
            result.onSuccess { list ->
                _foods.value = list.map {
                    BackendFood(
                        id = it.name, // id is not available from search mapping; fallback to name
                        name = it.name,
                        calories = it.nutriments.calories,
                        protein = it.nutriments.protein,
                        carbs = it.nutriments.carbohydrates,
                        fat = it.nutriments.fat
                    )
                }
            }.onFailure {
                _foods.value = emptyList()
            }
        }
    }

    fun createFood(
        name: String,
        calories: Float,
        protein: Float?,
        carbs: Float?,
        fat: Float?,
        sugar: Float? = null,
        sfat: Float? = null,
        freeSugar: Float? = null,
        fibres: Float? = null,
        sel: Float? = null,
        alcool: Float? = null,
        onDone: () -> Unit
    ) {
        viewModelScope.launch {
            repository.createFood(name, calories, protein, carbs, fat, sugar, sfat, freeSugar, fibres, sel, alcool)
            onDone()
        }
    }

    fun deleteFood(foodId: String) {
        viewModelScope.launch {
            repository.deleteFood(foodId)
        }
    }
}