# analys_trophies_PCA

17Lands.comのデータをを使い７勝デッキをまとめ PCAし非階層型クラスタリングして2次元散布図を作るコードです。
これにより色の強弱や色の関係性の高いアーキカラーを見つけることができます。

使用するデータ https://www.17lands.com/public_datasets Game Dataデータ（各自サイトからDL）

或いは https://www.17lands.com/trophies からデッキリストを集めて結合した自前のデータ(各自で用意する。）

別途用意するもの（htmlファイルで画像を表示するのに必要）
・imgフォルダ内のカード画像
・name_list.tsv
　(データのカード名とimg内のカード画像ファイル名を紐づけしています。)


カードセット毎に行う行為について
■python内の書き換えについて
CSVファイル等の書き換え要素が発生するものはdef main()より上記に記載 セット毎にデータファイル名が違うので各セット毎にファイル名は書き換えする必要あり。 出来るだけ１ファイル内で書き換えを完結したかったので「除外する土地」などの除外カードもpyファイル内に記載してある。

全てのアーキカラーPCA非階層クラスタリング
![ALL_Archetype_PCA](https://github.com/ManaBurnSaito/analys_trophies_PCA/assets/139425458/4c887a82-246b-4e7a-ba84-0474d1641611)

アーキカラー毎のPCA非階層クラスタリング
![BR](https://github.com/ManaBurnSaito/analys_trophies_PCA/assets/139425458/934bf9fb-1274-4b45-8e7f-f1756b84403a)

