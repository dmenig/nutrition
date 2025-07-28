package com.nutrition.app.ui.customfood

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.nutrition.app.data.model.CustomFood
import com.nutrition.app.data.repository.CustomFoodRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class CustomFoodViewModel @Inject constructor(
    private val repository: CustomFoodRepository
) : ViewModel() {

    private val _customFoods = MutableStateFlow<List<CustomFood>>(emptyList())
    val customFoods: StateFlow<List<CustomFood>> = _customFoods.asStateFlow()

    init {
        viewModelScope.launch {
            repository.getAllCustomFoods().collect { foods ->
                _customFoods.value = foods
            }
        }
    }

    fun addCustomFood(customFood: CustomFood) {
        viewModelScope.launch {
            repository.insertCustomFood(customFood)
        }
    }

    fun deleteCustomFood(customFood: CustomFood) {
        viewModelScope.launch {
            repository.deleteCustomFood(customFood)
        }
    }
}