# [Day 24]由淺入深！介紹更多MealPy的API (2/2)

- Day: 24
- Date: 2024-09-30 00:01:06
- Author: golucky_sir
- Source: https://ithelp.ithome.com.tw/articles/10360827
- Series: https://ithelp.ithome.com.tw/2020-12th-ironman/articles/7610
- Series Title: 調整AI超參數好煩躁？來試試看最佳化演算法吧！

## 前言

[昨天](https://ithelp.ithome.com.tw/articles/10360297)介紹了MealPy的一些進階功能，包括繪圖與多目標最佳化，今天要來向各位介紹自定義問題等技巧，自定義問題在使用比較複雜的問題例如深度學習模型最佳化等應用就很適合。  
現在就來看看如何使用MealPy來實作這些功能吧~

## MealPy自定義最佳化問題

在MealPy自定義**比較複雜的**問題時也會建立一個額外類別，並在其中定義所需的輸入參數，就不單純只是建立副程式，不過具體要如何設定還是要看當下的問題以及個人的喜好等。  
在MealPy中設定自定義問題時我們需要繼承`Problem`類別：

    from mealpy import Problem
    class MyProblem(Problem):
        pass

以下我將以**使用[黏菌最佳化演算法(Slime Mould Algorithm, SMA)](https://doi.org/10.1016/j.future.2020.03.055)對Griewank Function進行最佳化**為例為各位簡單展示如何使用MealPy自定義最佳化問題。

1.  **定義求解類別以及初始化**：這部分主要要注意繼承類別，以及初始化。

    - 在MealPy中設定自定義問題時我們需要繼承`Problem`類別：

    <!-- -->

        from mealpy import Problem
        class MyProblem(Problem):
            pass

    - 接著我們要進行初始化，初始化有一些東西需要設定，基本上就是把問題字典中的key作為類別初始化需要設定的參數，在呼叫問題類別時就直接把這些部分都設定好了。

    <!-- -->

        def __init__(self, minmax, bounds=None, name="", **kwargs):  # 可以根據需求自定義其他參數
            self.name = name
            # 可以設定其他參數，或者進行其他初始化
            super().__init__(bounds, minmax, **kwargs)

2.  **定義適應值計算方式**：這部分就是以往我們定義適應值的計算方式了，不過副程式名稱要注意要繼承自`Problem`類別，該方法名稱為`def obj_func(self, x):`，完整程式如下。

        def obj_func(self, x):
            # 其他處理，將輸入x轉換為numpy陣列的形式
            x = np.array(x)
            # 定義目標函數的計算
            i = np.arange(1, len(x) + 1)
            x1 = np.sum(x ** 2 / 4000)
            x2 = -np.prod(np.cos(x / np.sqrt(i)))
            return x1 + x2 + 1  # 定義回傳適應值

3.  **定義其他功能**：最後就可以定義其他功能了，看有沒有需要針對問題或者收集實驗資料等擴充一些功能都可以設定。  
    假設我想要定義一個方法是**可以繪製出收斂過程圖的話**，那可以這樣做：

        def plot_result(self):
            optimizer.history.save_global_objectives_chart(filename="result/global objectives chart")

4.  **求解問題**：這部分就只是呼叫自定義的問題，前幾天的教學是要設定問題字典作為設定問題中一些參數的方式，如果使用自定義問題的話，就可以直接將這些參數在初始化問題時一併傳入，除此之外也可以設定其他初始化參數，在面對比較複雜的問題會比較好用。最後就是定義演算法並直接求解即可。  
    以下為這部分的程式範例。

        # 設定問題，問題中的設定會作為初始化參數傳遞進去。
        problem = GriewankFunction(bounds=FloatVar(lb=[-600] * 10, ub=[600] * 10),
                                   name="Griewank Function", minmax="min")
        # 求解問題
        optimizer = SMA.OriginalSMA(epoch=30, pop_size=50, pr=0.03)
        optimizer.solve(problem=problem)
        # 繪製收斂曲線
        problem.plot_result()

## 儲存 & 載入最佳化試驗

接著我們要來討論如何儲存或者載入最佳化的試驗，不過目前根據官方文檔說明，目前似乎還沒提供方法可以讓原本的試驗可以繼續進行的能力。  
現階段保存模型之後僅提供調用資料確認試驗的細節等等，不像之前介紹的Optuna一樣可以繼承試驗來繼續最佳化。  
要使用MealPy的儲存模型功能需要額外匯入一個模組：`from mealpy.utils import io`。接著只需要使用`io.save_model()`方法就可以儲存模型了；之後要調用模型使用`io.load_model()`方法就好！  
完整程式如下：

    from mealpy.utils import io  # 儲存、讀取演算法時使用

    # 主程式，定義演算法與問題並求解

    # 儲存最佳化試驗
    io.save_model(optimizer, "模型儲存路徑.pkl")  # 指定儲存路徑
    # 讀取最佳化試驗，並且調用該次試驗的最佳解資訊
    optimizer_read = io.load_model("模型載入路徑.pkl")  # 指定載入路徑
    # 調用該次試驗時的最佳解資料
    print(f"Best solution: {optimizer_read.g_best.solution}, Best fitness: {optimizer_read.g_best.target.fitness}")

## 結語

今天介紹了如何使用MealPy自定義問題類別，這在之後用於深度學習模型的最佳化時通常可以更好的分割程式碼，將程式區分出不同部分來讓提高程式碼的可讀性。  
另外也向各位分享了儲存與載入最佳化試驗的方式，可以提供各位保存自己的試驗，在之後可以調用出來分析，就不怕試驗的過程遺失導致要重新實驗了。  
接下來我將介紹如何使用MealPy來進行模型參數的最佳化，基本上跟Optuna差不多，只是使用MealPy可以再選擇更多啟發式演算法來實驗看看，這些演算法都沒有說誰是絕對優秀的，所以多方比較有時候才能得到比較完整且令人滿意的內容喔~

## 附錄：完整程式

    from mealpy import FloatVar, Problem, SMA
    from mealpy.utils import io  # 儲存、讀取演算法時使用
    import numpy as np

    class GriewankFunction(Problem):
        def __init__(self, minmax, bounds=None, name="", **kwargs):  # 可以根據需求自定義其他參數
            self.name = name
            # 可以設定其他參數，或者進行其他初始化
            super().__init__(bounds, minmax, **kwargs)

        def obj_func(self, x):
            # 其他處理，將輸入x轉換為numpy陣列的形式
            x = np.array(x)
            # 定義目標函數的計算
            i = np.arange(1, len(x) + 1)
            x1 = np.sum(x ** 2 / 4000)
            x2 = -np.prod(np.cos(x / np.sqrt(i)))
            return x1 + x2 + 1  # 定義回傳適應值

        def plot_result(self):
            optimizer.history.save_global_objectives_chart(filename="result/global objectives chart")

    # 設定問題，問題中的設定會作為初始化參數傳遞進去。
    problem = GriewankFunction(bounds=FloatVar(lb=[-600] * 10, ub=[600] * 10),
                               name="Griewank Function", minmax="min")
    # 求解問題
    optimizer = SMA.OriginalSMA(epoch=30, pop_size=50, pr=0.03)
    optimizer.solve(problem=problem)
    # 繪製收斂曲線
    problem.plot_result()

    # 儲存最佳化試驗
    io.save_model(optimizer, "result/SMA optimizer.pkl")  # 指定儲存路徑
    # 讀取最佳化試驗，並且調用該次試驗的最佳解資訊
    optimizer_read = io.load_model("result/SMA optimizer.pkl")  # 指定載入路徑
    # 調用該次試驗時的最佳解資料
    print(f"Best solution: {optimizer_read.g_best.solution}, Best fitness: {optimizer_read.g_best.target.fitness}")
