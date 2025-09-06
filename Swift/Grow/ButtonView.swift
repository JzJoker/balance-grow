import SwiftUI

struct PrimaryButton: View {
    let title: String
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Text(title)
                .padding(.vertical)
                .frame(maxWidth: .infinity)
                .background(Color.primary)
                .foregroundColor(.white)
                .cornerRadius(.infinity)
        }
    }
}

struct SecondaryButton: View{
    let title:String
    let action:()->Void
    
    var body: some View {
        Button(action:action){
            Text(title)
                .padding(.vertical, 16)
                .frame(maxWidth: .infinity)
                .background(Color(.systemGray6))
                .foregroundColor(.primary)
                .cornerRadius(.infinity)
        }
    }
}

#Preview {
    HStack(spacing: 16){
            PrimaryButton(title: "Hello World", action: {})
            SecondaryButton(title: "Hello", action: {})
    }.padding()
}
