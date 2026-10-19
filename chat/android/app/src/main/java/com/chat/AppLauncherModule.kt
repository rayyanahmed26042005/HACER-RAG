package com.chat

import android.content.Intent
import android.content.pm.PackageManager
import android.util.Log
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.Promise

class AppLauncherModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    override fun getName(): String {
        return "AppLauncherModule"
    }

    @ReactMethod
    fun isAppInstalled(packageName: String, promise: Promise) {
        try {
            val pm = reactContext.packageManager
            // Primary check: Look for a launch intent
            val launchIntent = pm.getLaunchIntentForPackage(packageName)
            if (launchIntent != null) {
                Log.d("AppLauncherModule", "Package $packageName has a launch intent")
                promise.resolve(true)
                return
            }
            // Fallback: Check package info without GET_ACTIVITIES
            pm.getPackageInfo(packageName, 0)
            Log.d("AppLauncherModule", "Package $packageName found via getPackageInfo")
            promise.resolve(true)
        } catch (e: PackageManager.NameNotFoundException) {
            Log.d("AppLauncherModule", "Package $packageName not found")
            promise.resolve(false)
        } catch (e: Exception) {
            Log.e("AppLauncherModule", "Error checking package $packageName: ${e.message}")
            promise.reject("CHECK_ERROR", "Error checking app installation: ${e.message}")
        }
    }

    @ReactMethod
    fun launchApp(packageName: String, promise: Promise) {
        try {
            val pm = reactContext.packageManager
            val launchIntent = pm.getLaunchIntentForPackage(packageName)
            if (launchIntent != null) {
                launchIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                reactContext.startActivity(launchIntent)
                Log.d("AppLauncherModule", "Launched app with package $packageName")
                promise.resolve(true)
            } else {
                Log.w("AppLauncherModule", "No launch intent for package $packageName")
                promise.reject("LAUNCH_ERROR", "App not found or cannot be launched: $packageName")
            }
        } catch (e: Exception) {
            Log.e("AppLauncherModule", "Failed to launch app $packageName: ${e.message}")
            promise.reject("LAUNCH_ERROR", "Failed to launch app: ${e.message}")
        }
    }
}