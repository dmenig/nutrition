package com.nutrition.app.ui.foodentry

import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import org.junit.Rule
import org.junit.Test
import java.util.Date

class FoodEntryFormTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun saveButton_isDisplayed() {
        composeTestRule.setContent {
            FoodEntryForm(
                onSave = { _, _, _ -> },
                onCancel = { }
            )
        }

        composeTestRule.onNodeWithText("Save").assertIsDisplayed()
    }

    @Test
    fun saveButton_triggersOnSaveCallback() {
        var saveClicked = false
        composeTestRule.setContent {
            FoodEntryForm(
                onSave = { _, _, _ -> saveClicked = true },
                onCancel = { }
            )
        }

        composeTestRule.onNodeWithText("Save").performClick()
        assert(saveClicked)
    }

    @Test
    fun cancelButton_isDisplayed() {
        composeTestRule.setContent {
            FoodEntryForm(
                onSave = { _, _, _ -> },
                onCancel = { }
            )
        }

        composeTestRule.onNodeWithText("Cancel").assertIsDisplayed()
    }

    @Test
    fun cancelButton_triggersOnCancelCallback() {
        var cancelClicked = false
        composeTestRule.setContent {
            FoodEntryForm(
                onSave = { _, _, _ -> },
                onCancel = { cancelClicked = true }
            )
        }

        composeTestRule.onNodeWithText("Cancel").performClick()
        assert(cancelClicked)
    }
}