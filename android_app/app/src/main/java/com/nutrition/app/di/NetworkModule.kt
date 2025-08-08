package com.nutrition.app.di

import com.nutrition.app.data.remote.NutritionApi
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import retrofit2.Retrofit
import javax.inject.Named
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object NetworkModule {

    @Provides
    @Singleton
    fun provideNutritionApi(@Named("NutritionApi") retrofit: Retrofit): NutritionApi =
        retrofit.create(NutritionApi::class.java)
}