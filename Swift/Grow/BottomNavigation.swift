//
//  BottomNavigation.swift
//  Grow
//
//  Created by Junheng Zheng on 9/6/25.
//

import SwiftUI

struct navigationTab:View{
    var icon:Image
    var text:String
    var isActive: Bool = false
    
    var body:some View{
        VStack(spacing:4){
            icon.foregroundStyle(Color.purple80)
            Text(text).font(.system(size:12)).foregroundColor(Color.purple80)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(isActive ? Color.purple10 : Color.white)
        .clipShape(
            .rect(
                topLeadingRadius: 12,
                bottomLeadingRadius: 0,
                bottomTrailingRadius: 0,
                topTrailingRadius: 12
            )
        )
    }
    
}
struct bottomNavigation:View {
    @State private var selectedIndex:Int = 0
    
    var body: some View {
        HStack(spacing:12){
            navigationTab(icon: Image(systemName: "house.fill"), text: "Home", isActive: selectedIndex == 0).onTapGesture { selectedIndex = 0 }
            navigationTab(icon: Image(systemName: "pill.fill"), text: "Routine", isActive: selectedIndex == 1).onTapGesture { selectedIndex = 1 }
            VStack(spacing:0){
                Image(systemName: "camera.fill").foregroundColor(Color.white).font(.system(size:24))
            }.frame(width:74, height:74).background(Color.primary).cornerRadius(.infinity)
            navigationTab(icon: Image(systemName: "barcode"), text: "Scans", isActive: selectedIndex == 3).onTapGesture { selectedIndex = 3 }
            navigationTab(icon: Image(systemName: "newspaper"), text: "Articles", isActive: selectedIndex == 4).onTapGesture { selectedIndex = 4 }

        }.frame(height:74).background(Color.white)
    }
}


#Preview {
    VStack(){
        Spacer()
        bottomNavigation()
    }.frame(maxHeight: .infinity)
}
