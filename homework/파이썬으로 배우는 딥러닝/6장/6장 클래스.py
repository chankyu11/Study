# # p.167
# # 클래스

# class MyProduct:
#     # 생성자를 수정하세요
#     def __init__(self, name, price, stock):
#         # 인수를 멤버에 저장하세요
#         self.name = name
#         self.price = price
#         self.stock = stock
#         self.sales = 0

# product_1 = MyProduct("cake", 500, 20)
# print(product_1.stock)
# print(product_1.name)
# print(product_1.price)

class MyProduct:
    def __init__(self, name, price, stock):
        self.name = name
        self.price = price
        self.stock = stock
        self.sales = 0
    def buy_up(self, n):
        self.stock += n
    def sell(self, n):
        self.stock -= n
        self.sales += n * self.price
    def summary(self):
        message = "called summay().\n name: " + self.name + \
        "\n price: " + str(self.price) + \
        "\n stock: " + str(self.stock) + \
        "\n sales: " + str(self.sales)
        print(message) 


class MyProduct:
    def __init__(self, name, price, stock):
        self.name = name
        self.price = price
        self.stock = stock
        self.sales = 0

    # 문자열과 "자신의 방법"과 "자신의 구성원"을 연결하여 출력하세요
    def summary(self):
        message = "called summary()." + \
        "\n name: " + self.get_name() + \
        "\n price: " + str(self.price) + \
        "\n stock: " + str(self.stock) + \
        "\n sales: " + str(self.sales)
        print(message)
    
    # return name = get_name()
    def get_name(self):
        return self.name

    # 인수 - price
    def discount(self, n):
        self.price -= n

product_2 = MyProduct("phone", 30000, 100)

# n - 5,000
# product_2.discount(5000)

# print(product_2.summary)
# product_2.summary()

# MyProduct 클래스를 상속하는 MyProductSalesTax을 정의
class MyProductSalesTax(MyProduct):
    
    # MyProductSalesTax는 생성자의 네 번째 인수가 소비세 비율을 받습니다
    def __init__(self, name, price, stock, tax_rate):
        # super()를 사용하면 부모 클래스의 메서드를 호출 가능
        # 여기서는 MyProduct 클래스의 생성자를 호출합니다
        super().__init__(name, price, stock)
        self.tax_rate = tax_rate

    # MyProductSalesTax에서 MyProduct의 get_name을 재정의(오버라이드)합니다
    def get_name(self):
        return self.name + "(세금 포함)"

    # MyProductSalesTax에서 get_price_with_tax를 新加入
    def get_price_with_tax(self):
        return int(self.price * (1 + self.tax_rate))
    def summary(self):
        message = "called summary().\n name: " + self.get_name() + \
        "\n price: " + str(self.get_price_with_tax()+0) + \
        "\n stock: " + str(self.stock) + \
        "\n sales: " + str(self.sales)
        print(message) 

product_3 = MyProductSalesTax("phone", 30000, 100, 0.1)
print(product_3.get_name())
print(product_3.get_price_with_tax())
product_3.summary()