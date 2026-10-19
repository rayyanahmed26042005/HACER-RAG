package com.rtnmyalarm

import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.ViewManager
import com.facebook.react.module.model.ReactModuleInfo
import com.facebook.react.module.model.ReactModuleInfoProvider
import com.facebook.react.TurboReactPackage

class MyAlarmPackage : TurboReactPackage() {

    override fun getModule(name: String, reactContext: ReactApplicationContext): NativeModule? {
        return if (name == MyAlarmModule.NAME) {
            MyAlarmModule(reactContext)
        } else {
            null
        }
    }

   override fun getReactModuleInfoProvider(): ReactModuleInfoProvider {
    return ReactModuleInfoProvider {
        val moduleInfos = HashMap<String, ReactModuleInfo>()
        moduleInfos[MyAlarmModule.NAME] = ReactModuleInfo(
            MyAlarmModule.NAME,
            MyAlarmModule.NAME,
            false, // canOverrideExistingModule
            false, // needsEagerInit
            false, // hasConstants
            false  // isCxxModule (corrected)
        )
        moduleInfos
    }
}

    override fun createViewManagers(
        reactContext: ReactApplicationContext
    ): List<ViewManager<*, *>> {
        return emptyList()
    }
}
