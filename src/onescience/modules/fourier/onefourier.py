
class OneFourier(nn.Module):
    def __init__(self, style="FNOFourider"):
        if style == "FNOFourider":
            self.style = FNOFourier()
        else:
            raise NotImplementedError