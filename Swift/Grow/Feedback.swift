//
//  Feedback.swift
//  Grow
//
//  Created by Junheng Zheng on 9/6/25.
//

import SwiftUI

struct Feedback:View{
    var text:String
    var learnmore:String
    @State private var showSheet = false
    
    var body: some View{
        VStack{
            HStack{
                Rectangle().fill(Color.red).frame(width: 38, height: 38).cornerRadius(.infinity)
                Text(text).lineLimit(2)
            }
            Button("Learn More"){
                showSheet.toggle()
            }.sheet(isPresented: $showSheet) {
                HStack{
                    Text("Hello")
                }.background(Color.red).frame(maxWidth:.infinity)
            }.frame(maxWidth: .infinity)
        }.frame(maxWidth: .infinity)
        
    }
}

#Preview {
    Feedback(text: "Hello", learnmore: "Learn More")
}
