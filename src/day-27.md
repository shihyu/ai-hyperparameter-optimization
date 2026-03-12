# [Day 27]MealPy的更多應用，最佳化生成對抗網路(GAN)(1/2)

- Day: 27
- Date: 2024-10-03 00:01:52
- Author: golucky_sir
- Source: https://ithelp.ithome.com.tw/articles/10362517
- Series: https://ithelp.ithome.com.tw/2020-12th-ironman/articles/7610
- Series Title: 調整AI超參數好煩躁？來試試看最佳化演算法吧！

## 前言

今天要來介紹一下如何使用MealPy再進行生成對抗網路的最佳化，前幾天我有先預防性的跑一次程式了XD，跑了非常非常久。接下來和之前一樣我將帶各位在原有DCGAN的基礎上再進階套用一層最佳化演算法~只是之前使用Optuna，這次使用MealPy。

> 今天的內容與[第20天](https://ithelp.ithome.com.tw/articles/10358474)介紹的差不多，但是一樣是改成使用MealPy並使用[第22天](https://ithelp.ithome.com.tw/articles/10359653)介紹的流程來開發程式。相信在經過了幾次實戰操作後各位應該都有得到要領了，也希望各位可以根據此基礎流程去延伸、改良成最適合自己開發步調的流程。

## DCGAN程式碼

各位可以直接到我去年介紹的文章中，我有附上[DCGAN完整程式碼(最底下附錄)](https://ithelp.ithome.com.tw/articles/10318972)，程式碼的詳細說明可以參考去年的文章。  
今天會挑選部分程式來進行修改，底下也會附上完整修改過的程式碼喔。之前執行時有經過一些手動調整與最佳化，所以才能跑出比較能看的結果，今天就來嘗試進一步的最佳化GAN模型吧！

> DCGAN的程式碼使用去年的程式碼並獨立分成一個檔案DCGAN.py，在主程式直接調用會讓程式碼比較簡潔，這段程式會附在最底下。  
> 程式碼的完整訓練結果在去年文章我有詳細的說明喔！

## 構思問題

這次也是因篇幅所以不會新增太多其他的功能，儲存模型與訓練資料等部分在此就不過多敘述了，各位可以看看我[去年的文章](https://ithelp.ithome.com.tw/articles/10318972)來理解這些部分的細節。

> 為了要正確執行結果並展示內容，以及要顧及到我電腦的效能，所以範例的一些內容會被簡化(例如**DCGAN的訓練次數**，以及**最佳化演算法迭代數**跟**群數量**)，基本上顧慮到節省效能所以模型訓練結果應該不會太好，所以在實務應用上請依據實際需求去更改，通常上述簡化的內容都要再增加許多。

| 5W1H  | 規劃內容                                                                                                                                                                              |
|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Why   | 最佳化DCGAN模型，目標為適應值越高越好                                                                                                                                                 |
| What  | 最佳化問題是DCGAN的圖片生成，以PSNR與SSIM指標作為適應值                                                                                                                               |
| Who   | 預計對DCGAN中的生成器與判別器的**學習率**、**第一層卷積網路的神經元數量**、**卷積層的卷積核大小**以及**判別器中LeakyReLU的斜率alpha**進行最佳化                                       |
| Where | 學習率設定在0.00001~0.001、生成器第一層神經元數量從\[32, 64, 128, 256\]中搜尋；生成器第一層神經元數量從\[64, 128, 256, 512\]中搜尋、卷積層的卷積核大小為1~5、alpha設定0.01~0.5。      |
| When  | 測試跑完一次程式後確定沒問題即可著手進行最佳化。                                                                                                                                      |
| How   | 使用MealPy的[黏菌最佳化演算法(Slime Mould Algorithm, SMA)](https://mealpy.readthedocs.io/en/latest/pages/models/mealpy.bio_based.html?highlight=SMA#mealpy.bio_based.SMA.OriginalSMA) |

## 實現MealPy最佳化

我們來根據之前介紹的流程，將程式逐步的撰寫出來，經過幾天的分享，如果都有再自己動手實作的話應該對這些流程也感到熟悉了吧！  
基本上我們在去年將DCGAN.py的功能寫得蠻完整的了，接下來就只需要新增類別並調用方法、定義搜索空間等就好了！在第20天的文章中我也有使用Optuna來進行最佳化，流程上相差不了太多。

1.  **定義目標函數**：首先需要新增自定義問題的類別，類別需要繼承`Problem`，所以一開始的程式就是：  
    除此之外也可以順便`import`其他東西。

        from mealpy import Problem
        class Optimize_DCGAN(Problem):

    - **初始化方法**：接著要來進行類別的初始化。  
      在上面的表格中有提到生成器與判別器的第一層神經元會從一個離散的串列搜索空間中取值出來使用，所以初始化我們需要額外定義這兩個部分(`self.Generator_first_layer_unit_lst`和`self.Discriminator_first_layer_unit_lst`)。

          def __init__(self, minmax, bounds=None, name="", **kwargs):  # 可以根據需求自定義其他參數
              self.name = name
              # 設定其他參數，或者進行其他初始化
              self.Generator_first_layer_unit_lst = [32, 64, 128, 256]  # 生成器的第一隱藏層層數搜索空間串列
              self.Discriminator_first_layer_unit_lst = [64, 128, 256, 512]  # 判別器的第一隱藏層層數搜索空間串列

              super().__init__(bounds, minmax, **kwargs)

    - **定義目標函數中的計算**：這裡就是直接定義目標函數的計算，我會將解進行初步的處理，並輸入至`DCGAN`類別中進行初始化、訓練模型。  
      在這幾天跑程式時我又發現一項MealPy的小瑕疵，在使用`IntegerVar(lb=x, ub=y)`進行最佳化搜索時，實際的搜索空間會是從**x-0.5到y+0.5**，我個人表示很問號XD，明明就是整數資料的搜索了，為何取值範圍會是浮點數，而且輸出結果同樣也是浮點數資料型態。  
      所以當我設定卷積核(`g_k`和`d_k`)的大小時，我設定`IntegerVar(lb=1, ub=5)`，卻跳出錯誤說核大小不可為0，這才發現MealPy某次試驗取值取了0.5後轉為整數型態變成0，才導致錯誤發生，所以各位在使用時需要再注意一下這個小錯誤。  
      以下為這個部分的完整程式，有一些說明都於程式當中的註解中說明完畢：

          def obj_func(self, x):
              """
              DCGAN 網路訓練的最佳化。
              """
              # 為了使程式碼可讀性提高所以我先初步處理輸入解
              generator_lr = x[0]
              discriminator_lr = x[1]
              # 使用索引值從搜索空間中選出特定的元素，因為MealPy的一些缺陷導致輸出會被統一成浮點數型態
              # 這缺陷實際上是因為帶入解釋numpy array格式，所以dtype會被統一成浮點數，導致後續需要自己再處理
              g_first_layer_unit = self.Generator_first_layer_unit_lst[int(x[2])]
              d_first_layer_unit = self.Discriminator_first_layer_unit_lst[int(x[3])]
              g_k = int(x[4])
              d_k = int(x[5])
              alpha = x[6]
              # 將解帶入DCGAN類別中並進行訓練
              gan = DCGAN(generator_lr=generator_lr,
                          discriminator_lr=discriminator_lr,
                          g_first_layer_unit=g_first_layer_unit,
                          d_first_layer_unit=d_first_layer_unit,
                          g_k=g_k,
                          d_k=d_k,
                          alpha=alpha)
              # 為了使訓練速度加快，所以訓練次數設定很低，原則上會訓練大約20000次。
              gan.train(epochs=5000, batch_size=128)  
              # 定義回傳適應值
              fitness = gan.calculate_finess_value()
              del gan  # 刪除掉gan這個類別，釋放一些記憶體空間。
              return fitness

    - **完善其他功能**：這部分各位可以根據需求去新增當訓練進行時或者完成後都生成一批的圖片來觀察訓練的結果情形。為了降低我電腦的負擔這部分就沒有新增，不過`DCGAN.py`中都有相關的功能，去年介紹程式時也都有詳細說明程式，若有GAN實作上的問題歡迎再去看看我去年的系列文章。

    - **定義回傳適應值**：最後就是回傳適應值了，這部分去年並無實作，所以我是直接在DCGAN.py中新增這項功能，在主程式中回傳適應值時只需要呼叫方法就可以了。  
      以下是在DCGAN.py中定義的適應值計算的部分，各位可以看看註解的說明，以理解適應值設定的方式，設定並沒有絕對正確的方式，可以根據需求設定~

          # 在DCGAN.py中新增的方法
          def calculate_finess_value(self):
              noise = np.random.normal(0, 1, (50, 100))
              # 將資料格式統一成float32否則計算指標可能會出現錯誤
              gen_imgs = self.generator.predict(noise).astype('float32')
              x_train = self.load_data()[:50].astype('float32')
              # PSNR越高越好，PSNR通常數值比較高，大約0~50左右
              psnr = np.mean(tf.image.psnr(gen_imgs, x_train, max_val=1))
              # SSIM越接近1越好，SSIM數值比較低，大約0~1左右，有時候會為比較小的負數
              ssim = np.mean(tf.image.ssim(gen_imgs, x_train, max_val=1))
              # 為了平衡將SSIM的結果*50再加上PSNR再除以2作為適應值回傳
              return (psnr + 50*ssim)/2

      在主程式中因為訓練完成後為了節省**記憶體空間**所以我會將整個`DCGAN`類別刪除，所以會額外使用一個變數儲存適應值並回傳，在主程式的呼叫如下(就是目標函數`obj_func()`中的最後部分)：

          # 定義回傳適應值
          fitness = gan.calculate_finess_value()
          del gan  # 刪除掉gan這個類別，釋放一些記憶體空間。
          return fitness

2.  **定義試驗**：接著就來定義試驗吧，以下有一些注意事項也要留意一下。

    - **選擇一個最佳化演算法**：今天我們使用[黏菌最佳化演算法(Slime Mould Algorithm, SMA)](https://doi.org/10.1016/j.future.2020.03.055)來尋找最佳解喔，首先設定一下最佳化的演算法。  
      這個部份其實大家應該都熟悉了啦，這幾天的例子都使用這個演算法，也可以根據[官方文檔的說明](https://mealpy.readthedocs.io/en/latest/pages/models/mealpy.html)使用看看其他演算法來進行最佳化喔。另外為了節省程式執行的時間所以我`epoch`和`pop_size`的值都設定的很低，通常會建議值再設定的更高一些，尤其是`epoch`可以設定30次以上，但就要小心程式執行時間會非常久了。

      > 程式執行時間一長就會建議各位在一定時間過後進行儲存並備份結果，也可以設定提早結束的條件等都可以~

          optimizer = SMA.OriginalSMA(epoch=10, pop_size=5, pr=0.03)

    - **設定要帶入目標函數的變數**：接著來定義問題的搜索空間以及最佳化目標，基本上設定沒什麼變動，可以參考上面提到的表格，最佳化目標是尋找PSNR+SSIM混和的**最大值**。  
      這邊要注意一下剛剛提到的MealPy的`IntegerVar`的一些瑕疵，所以設定值上需要特別注意一下，其他為了使程式碼較清晰，所以我**分開定義**所有超參數的搜索空間並**設定該參數的名稱**，希望可以幫助各位理解程式碼。

          problem = Optimize_DCGAN(bounds=[FloatVar(lb=0.00001, ub=0.001, name="generator_lr"),
                                           FloatVar(lb=0.00001, ub=0.001, name="discriminator_lr"),
                                           MixedSetVar(valid_sets=(32, 64, 128, 256), name="g_first_layer_unit"),
                                           MixedSetVar(valid_sets=(64, 128, 256, 512), name="d_first_layer_unit"),
                                           # 設定整數的話會值範圍會為lb-0.5 ~ ub+0.5，這會導致錯誤發生。
                                           IntegerVar(lb=2, ub=5, name='g_k'),
                                           IntegerVar(lb=2, ub=5, name='d_k'),
                                           FloatVar(lb=0.01, ub=0.5, name="alpha")],
                                   name="DCGAN_optimizer", minmax="max")

    - **根據其他需求進行設定**：這部分也沒有其他的需求，跳過~

3.  **執行試驗進行最佳化**：接下來就是進行試驗的最佳化了，也是老樣子使用一行程式就可以讓它動起來啦。

        optimizer.solve(problem=problem)

4.  **後續處理與分析**：這裡我們也是讓程式**輸出最佳解**、**最佳適應值**跟**繪製最佳化的收斂曲線**  
    **print最佳解**：程式碼如下，方式跟前幾天介紹的都一樣。

         ```python
         print(f"Best solution: {optimizer.g_best.solution}")
         print(f"Best fitness: {optimizer.g_best.target.fitness}")
         ```

    **產生視覺化圖表**：這部分也都是老樣子了，若各位有其他圖表想繪製的話，可以看看我在[第23天的文章](http://)中介紹的視覺化圖表部分，如果想改為讓PSNR與SSIM變成兩個目標並觀察收斂方式的話也可以參考那天文章中介紹的做法喔。

         ```python
         optimizer.history.save_global_objectives_chart(filename="result/global objectives chart")
         ```

5.  **分析最佳化結果**：這個部分就留到明天再討論吧！目前程式還在執行中，電腦整台都快燒起來了XD

## 結語

今天介紹了如何使用MealPy來進行DCGAN的最佳化方式。明天會來跟各位簡單的討論最佳化結果以及一些其他相關的東西。不過其他相關的這些東西好像在[第21天](https://ithelp.ithome.com.tw/articles/10359181)就討論的差不多了，希望明天不會沒梗orz。  
這次程式的DCGAN訓練次數又更短了，所以想當然而應該總成果應該並不會太好，不過具體如何還是要等明天程式跑完才知道囉。

## 附錄：完整程式(最佳化DCGAN)

    import numpy as np
    import optuna
    import matplotlib.pyplot as plt
    from DCGAN import DCGAN
    from mealpy import FloatVar, Problem, SMA, IntegerVar, MixedSetVar

    class Optimize_DCGAN(Problem):
        def __init__(self, minmax, bounds=None, name="", **kwargs):  # 可以根據需求自定義其他參數
            self.name = name
            # 設定其他參數，或者進行其他初始化
            self.Generator_first_layer_unit_lst = [32, 64, 128, 256]  # 生成器的第一隱藏層層數搜索空間串列
            self.Discriminator_first_layer_unit_lst = [64, 128, 256, 512]  # 判別器的第一隱藏層層數搜索空間串列

            super().__init__(bounds, minmax, **kwargs)

        def obj_func(self, x):
            """
            DCGAN 網路訓練的最佳化。
            """
            # 為了使程式碼可讀性提高所以我先初步處理輸入解
            generator_lr = x[0]
            discriminator_lr = x[1]
            # 使用索引值從搜索空間中選出特定的元素，因為MealPy的一些缺陷導致輸出會被統一成浮點數型態
            # 這缺陷實際上是因為帶入解釋numpy array格式，所以dtype會被統一成浮點數，導致後續需要自己再處理
            g_first_layer_unit = self.Generator_first_layer_unit_lst[int(x[2])]
            d_first_layer_unit = self.Discriminator_first_layer_unit_lst[int(x[3])]
            g_k = int(x[4])
            d_k = int(x[5])
            alpha = x[6]
            # 將解帶入DCGAN類別中並進行訓練
            gan = DCGAN(generator_lr=generator_lr,
                        discriminator_lr=discriminator_lr,
                        g_first_layer_unit=g_first_layer_unit,
                        d_first_layer_unit=d_first_layer_unit,
                        g_k=g_k,
                        d_k=d_k,
                        alpha=alpha)
            # 為了使訓練速度加快，所以訓練次數設定很低，原則上會訓練大約20000次。
            gan.train(epochs=5000, batch_size=128)
            # 定義回傳適應值
            fitness = gan.calculate_finess_value()
            del gan  # 刪除掉gan這個類別，釋放一些記憶體空間。
            return fitness

    if __name__ == '__main__':
        # 新增最佳化試驗
        # 設定問題，問題中的設定會作為初始化參數傳遞進去。
        # 建立最佳化DCGAN問題
        problem = Optimize_DCGAN(bounds=[FloatVar(lb=0.00001, ub=0.001, name="generator_lr"),
                                         FloatVar(lb=0.00001, ub=0.001, name="discriminator_lr"),
                                         MixedSetVar(valid_sets=(32, 64, 128, 256), name="g_first_layer_unit"),
                                         MixedSetVar(valid_sets=(64, 128, 256, 512), name="d_first_layer_unit"),
                                         # 設定整數的話會值範圍會為lb-0.5 ~ ub+0.5，這會導致錯誤發生。
                                         IntegerVar(lb=2, ub=5, name='g_k'),
                                         IntegerVar(lb=2, ub=5, name='d_k'),
                                         FloatVar(lb=0.01, ub=0.5, name="alpha")],
                                 name="DCGAN_optimizer", minmax="max")

        # 求解問題，考慮到程式執行時間就先用10個epoch就好了。
        optimizer = SMA.OriginalSMA(epoch=10, pop_size=5, pr=0.03)
        optimizer.solve(problem=problem)
        # 輸出歷史最佳解以及歷史最佳適應值
        print(f"Best solution: {optimizer.g_best.solution}")
        print(f"Best fitness: {optimizer.g_best.target.fitness}")
        # 繪製收斂曲線
        optimizer.history.save_global_objectives_chart(filename="result/global objectives chart")

## 附錄：完整程式(DCGAN.py)

    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU, Activation, Conv2DTranspose, Conv2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    import numpy as np
    import tensorflow as tf

    class DCGAN():
        def __init__(self,
                     generator_lr,
                     discriminator_lr,
                     g_first_layer_unit,
                     d_first_layer_unit,
                     g_k,
                     d_k,
                     alpha):
            """
            定義DCGAN的基本功能，包括定義模型，訓練模型，回傳適應值等。
            Args:
                generator_lr: 生成器學習率
                discriminator_lr: 判別器學習率
                g_first_layer_unit: 生成器第一層隱藏層卷積層的神經元數量，後續網路神經元數量都為第一層神經元數量之倍數。
                d_first_layer_unit: 判別器第一層隱藏層卷積層的神經元數量，後續網路神經元數量都為第一層神經元數量之倍數。
                g_k: 生成器卷積核大小
                d_k: 判別器卷積核大小
                alpha: 判別器LeakyReLU之負數部分斜率。
            """

            self.generator_lr = generator_lr
            self.discriminator_lr = discriminator_lr
            self.g_first_layer_unit = g_first_layer_unit
            self.d_first_layer_unit = d_first_layer_unit
            self.g_k = g_k
            self.d_k = d_k
            self.alpha = alpha

            self.discriminator = self.build_discriminator()
            self.generator = self.build_generator()
            self.adversarial = self.build_adversarialmodel()

            # Loss儲存在本範例不會用到，若有興趣可以自己實作後續損失分析等部分
            self.gloss = []
            self.dloss = []

        def load_data(self):
            (x_train, _), (_, _) = mnist.load_data()  # 底線是未被用到的資料，可忽略
            x_train = x_train / 255  # 正規化
            x_train = x_train.reshape((-1, 28, 28, 1))
            return x_train

        def build_generator(self):
            input_ = Input(shape=(100, ))
            x = Dense(7*7*32)(input_)
            x = Activation('relu')(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = Reshape((7, 7, 32))(x)
            # 設定第一層卷積網路的神經元數量以及卷積核大小
            x = Conv2DTranspose(self.g_first_layer_unit, kernel_size=self.g_k, strides=2, padding='same')(x)
            x = Activation('relu')(x)
            x = BatchNormalization(momentum=0.8)(x)
            # 設定第二層卷積網路的神經元數量，數量為第一層的2倍。以及卷積核大小
            x = Conv2DTranspose(self.g_first_layer_unit*2, kernel_size=self.g_k, strides=2, padding='same')(x)
            x = Activation('relu')(x)
            x = BatchNormalization(momentum=0.8)(x)
            out = Conv2DTranspose(1, kernel_size=self.g_k, strides=1, padding='same', activation='sigmoid')(x)

            model = Model(inputs=input_, outputs=out, name='Generator')
            model.summary()
            return model

        def build_discriminator(self):
            input_ = Input(shape = (28, 28, 1))
            # 設定第一層卷積網路的神經元數量以及卷積核大小
            x = Conv2D(self.d_first_layer_unit, kernel_size=self.d_k, strides=2, padding='same')(input_)
            x = LeakyReLU(alpha=self.alpha)(x)  # 設定LeakyReLU的斜率
            # 設定第二層卷積網路的神經元數量，數量為第一層的1/2倍，//為計算商數。以及卷積核大小
            x = Conv2D(self.d_first_layer_unit//2, kernel_size=self.d_k, strides=2, padding='same')(x)
            x = LeakyReLU(alpha=self.alpha)(x)  # 設定LeakyReLU的斜率
            # 設定第三層卷積網路的神經元數量，數量為第一層的1/4倍，//為計算商數。以及卷積核大小
            x = Conv2D(self.d_first_layer_unit//4, kernel_size=self.d_k, strides=1, padding='same')(x)
            x = LeakyReLU(alpha=self.alpha)(x)  # 設定LeakyReLU的斜率
            x = Flatten()(x)
            out = Dense(1, activation='sigmoid')(x)

            model = Model(inputs=input_, outputs=out, name='Discriminator')
            dis_optimizer = Adam(learning_rate=self.discriminator_lr , beta_1=0.5)
            model.compile(loss='binary_crossentropy',
                          optimizer=dis_optimizer,
                          metrics=['accuracy'])
            model.summary()
            return model
        def build_adversarialmodel(self):
            noise_input = Input(shape=(100, ))
            generator_sample = self.generator(noise_input)
            self.discriminator.trainable = False
            out = self.discriminator(generator_sample)
            model = Model(inputs=noise_input, outputs=out)

            adv_optimizer = Adam(learning_rate=self.generator_lr, beta_1=0.5)
            model.compile(loss='binary_crossentropy', optimizer=adv_optimizer)
            model.summary()
            return model

        def train(self, epochs, batch_size=128):
            # 準備訓練資料
            x_train = self.load_data()
            # 準備訓練的標籤，分為真實標籤與假標籤
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
            for epoch in range(epochs):
                # 隨機取一批次的資料用來訓練
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                imgs = x_train[idx]
                # 從常態分佈中採樣一段雜訊
                noise = np.random.normal(0, 1, (batch_size, 100))
                # 生成一批假圖片
                gen_imgs = self.generator.predict(noise)
                # 判別器訓練判斷真假圖片
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                #儲存鑑別器損失變化 索引值0為損失 索引值1為準確率
                self.dloss.append(d_loss[0])
                # 訓練生成器的生成能力
                noise = np.random.normal(0, 1, (batch_size, 100))
                g_loss = self.adversarial.train_on_batch(noise, valid)
                # 儲存生成器損失變化
                self.gloss.append(g_loss)
                # 將這一步的訓練資訊print出來
                print(f"Epoch:{epoch} [D loss: {d_loss[0]}, acc: {100 * d_loss[1]:.2f}] [G loss: {g_loss}]")

        def calculate_finess_value(self):
            noise = np.random.normal(0, 1, (50, 100))
            # 將資料格式統一成float32否則計算指標可能會出現錯誤
            gen_imgs = self.generator.predict(noise).astype('float32')
            x_train = self.load_data()[:50].astype('float32')
            # PSNR越高越好，PSNR通常數值比較高，大約0~50左右
            psnr = np.mean(tf.image.psnr(gen_imgs, x_train, max_val=1))
            # SSIM越接近1越好，SSIM數值比較低，大約0~1左右，有時候會為比較小的負數
            ssim = np.mean(tf.image.ssim(gen_imgs, x_train, max_val=1))
            # 為了平衡將SSIM的結果*50再加上PSNR再除以2作為適應值回傳
            return (psnr + 50*ssim)/2

    if __name__ == '__main__':
        # 執行一次看看程式有沒有問題
        gan = DCGAN(generator_lr=0.0002,discriminator_lr=0.0002, g_first_layer_unit=128,
                    d_first_layer_unit=128, g_k=2, d_k=2, alpha=0.2)
        gan.train(epochs=20000, batch_size=128)
