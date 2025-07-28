package com.nutrition.app.ui.foodentry

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.nutrition.app.data.local.dao.FrequentFoodItem
import com.nutrition.app.data.model.Product
import com.nutrition.app.data.repository.CustomFoodRepository
import com.nutrition.app.data.repository.FoodRepository
import com.nutrition.app.data.repository.OpenFoodFactsRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class FoodSearchViewModel @Inject constructor(
    private val customFoodRepository: CustomFoodRepository,
    private val openFoodFactsRepository: OpenFoodFactsRepository,
    private val foodRepository: FoodRepository
) : ViewModel() {

    private val _searchQuery = MutableStateFlow("")
    val searchQuery: StateFlow<String> = _searchQuery.asStateFlow()

    private val _searchResults = MutableStateFlow<List<Product>>(emptyList())
    val searchResults: StateFlow<List<Product>> = _searchResults.asStateFlow()

    private val _recentFoods = MutableStateFlow<List<String>>(emptyList())
    val recentFoods: StateFlow<List<String>> = _recentFoods.asStateFlow()

    private val _frequentFoods = MutableStateFlow<List<FrequentFoodItem>>(emptyList())
    val frequentFoods: StateFlow<List<FrequentFoodItem>> = _frequentFoods.asStateFlow()

    init {
        fetchRecentAndFrequentFoods()
        observeSearchQuery()
    }

    private fun observeSearchQuery() {
        viewModelScope.launch {
            searchQuery.debounce(300).collectLatest { query ->
                if (query.isNotBlank()) {
                    searchFoods(query)
                } else {
                    _searchResults.value = emptyList()
                }
            }
        }
    }

    fun onSearchQueryChanged(query: String) {
        _searchQuery.value = query
    }

    private fun searchFoods(query: String) {
        viewModelScope.launch {
            try {
                val customFoodsFlow = customFoodRepository.searchCustomFoods(query).map { list ->
                    list.map { Product(id = it.name, product_name = it.name, nutrients = null, image_front_url = null) }
                }
                val openFoodFactsFlow = flow {
                    emit(openFoodFactsRepository.searchFood(query)?.products ?: emptyList())
                }

                combine(customFoodsFlow, openFoodFactsFlow) { custom, off ->
                    custom + off
                }.collect { combined ->
                    _searchResults.value = combined
                }

            } catch (e: Exception) {
                // Handle error appropriately
            }
        }
    }

    private fun fetchRecentAndFrequentFoods() {
        viewModelScope.launch {
            foodRepository.getRecentFoodNames().collect {
                _recentFoods.value = it
            }
        }
        viewModelScope.launch {
            foodRepository.getFrequentFoodNames().collect {
                _frequentFoods.value = it
            }
        }
    }
}