package com.nutrition.app.ui.foodentry

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.nutrition.app.data.model.Product
import com.nutrition.app.data.repository.FoodRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class BarcodeScanViewModel @Inject constructor(
    private val foodRepository: FoodRepository
) : ViewModel() {

    private val _scannedFood = MutableStateFlow<Product?>(null)
    val scannedFood: StateFlow<Product?> = _scannedFood

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error

    fun fetchFoodDetails(barcode: String) {
        _isLoading.value = true
        _error.value = null
        viewModelScope.launch {
            try {
                val product = foodRepository.getFoodByBarcode(barcode)
                _scannedFood.value = product
            } catch (e: Exception) {
                _error.value = "Failed to fetch food details: ${e.message}"
            } finally {
                _isLoading.value = false
            }
        }
    }

    fun clearScannedFood() {
        _scannedFood.value = null
    }
}