package com.nutrition.app.di

import com.nutrition.app.data.PlotsApiService
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import retrofit2.Retrofit
import javax.inject.Singleton
import javax.inject.Named

@Module
@InstallIn(SingletonComponent::class)
object PlotsModule {

    @Provides
    @Singleton
    fun providePlotsApiService(@Named("NutritionApi") retrofit: Retrofit): PlotsApiService {
        return retrofit.create(PlotsApiService::class.java)
    }
}