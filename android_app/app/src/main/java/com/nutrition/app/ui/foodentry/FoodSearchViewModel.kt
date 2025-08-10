package com.nutrition.app.ui.foodentry

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.nutrition.app.data.NutritionRepository
import com.nutrition.app.data.model.Food
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.debounce
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class FoodSearchViewModel @Inject constructor(
    private val nutritionRepository: NutritionRepository
) : ViewModel() {

    private val _searchQuery = MutableStateFlow("")
    val searchQuery: StateFlow<String> = _searchQuery.asStateFlow()

    private val _searchResults = MutableStateFlow<List<Food>>(emptyList())
    val searchResults: StateFlow<List<Food>> = _searchResults.asStateFlow()

    private var searchJob: Job? = null

    init {
        observeSearchQuery()
    }

    private fun observeSearchQuery() {
        viewModelScope.launch {
            searchQuery
                .debounce(200L)
                .collectLatest { query: String ->
                    searchJob?.cancel()
                    if (query.isNotBlank() && query.length >= 2) { // Only search if query is 2 or more characters
                        searchJob = viewModelScope.launch {
                            autocompleteFoods(query)
                        }
                    } else {
                        _searchResults.value = emptyList()
                    }
                }
        }
    }

    fun onSearchQueryChanged(query: String) {
        _searchQuery.value = query
    }

    fun onSuggestionSelected(food: Food) {
        _searchQuery.value = food.name
        _searchResults.value = emptyList() // Clear search results after selection
    }

    private fun autocompleteFoods(query: String) {
        viewModelScope.launch {
            try {
                val result = nutritionRepository.autocompleteFoods(query)
                _searchResults.value = result.getOrDefault(emptyList())
            } catch (_: Exception) {
            }
        }
    }

}