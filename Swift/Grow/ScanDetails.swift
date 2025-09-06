//
//  ScanDetails.swift
//  Grow
//
//  Created by Junheng Zheng on 9/6/25.
//

import SwiftUI


struct ScanDetails: View {
    let header: String
    let text: String
    let number: Int
    var circleColor: Color {
        if number > 80 {
            return .green
        } else if number > 60 {
            return .yellow
        } else {
            return .red
        }
    }
    
    var body: some View {
        HStack(spacing:20) {
            VStack(alignment: .leading, spacing: 8) {
                Text(header)
                Text(text).font(.system(size:   12)).lineLimit(1).foregroundStyle(Color.gray10)
            }
            Spacer()
            HStack{
                Text("\(number)")
                (circleColor).frame(width:12, height:12).cornerRadius(.infinity)
            }
        }.frame(maxWidth: .infinity)
            .padding(.vertical, 20)
            .padding(.horizontal, 32)
            .overlay(
                Rectangle()
                    .frame(height: 1)
                    .foregroundColor(.gray5),
                alignment: .bottom
            )
    }
}





#Preview {
    VStack(spacing:0){
        ScanDetails(header: "Hairline", text: "This is a text to show the scan details on the actual hair line.", number: 100)
        ScanDetails(header: "Hairline", text: "This is a text to show the scan details on the actual hair line.", number: 70)
        ScanDetails(header: "Hairline", text: "This is a text to show the scan details on the actual hair line.", number: 30)
    }
}
