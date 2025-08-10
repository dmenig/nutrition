package com.nutrition.app.ui.dailylog

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import org.junit.Rule
import org.junit.Test
import org.mockito.kotlin.mock
import org.mockito.kotlin.verify

class DailyLogScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun addFoodButton_isDisplayed() {
        composeTestRule.setContent {
            DailyLogScreen(
                onNavigateToFoodEntry = {},
                onNavigateToSportEntry = {}
            )
        }

        composeTestRule.onNodeWithText("Add Food").assertIsDisplayed()
    }

    @Test
    fun addFoodButton_triggersNavigation() {
        var foodEntryNavigated = false
        composeTestRule.setContent {
            DailyLogScreen(
                onNavigateToFoodEntry = { foodEntryNavigated = true },
                onNavigateToSportEntry = {}
            )
        }

        composeTestRule.onNodeWithText("Add Food").performClick()
        assert(foodEntryNavigated)
    }

    @Test
    fun addSportButton_isDisplayed() {
        composeTestRule.setContent {
            DailyLogScreen(
                onNavigateToFoodEntry = {},
                onNavigateToSportEntry = {}
            )
        }

        composeTestRule.onNodeWithText("Add Sport").assertIsDisplayed()
    }

    @Test
    fun addSportButton_triggersNavigation() {
        var sportEntryNavigated = false
        composeTestRule.setContent {
            DailyLogScreen(
                onNavigateToFoodEntry = {},
                onNavigateToSportEntry = { sportEntryNavigated = true }
            )
        }

        composeTestRule.onNodeWithText("Add Sport").performClick()
        assert(sportEntryNavigated)
    }
}