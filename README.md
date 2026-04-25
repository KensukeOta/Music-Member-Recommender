# Music Member Recommender

## 概要

バンドメンバー募集サービスを想定し、ユーザーの属性情報および自己紹介文から相性の良いメンバーを推薦するシステムです。

ユーザーの楽器・ジャンル・活動地域・活動目的・活動頻度・活動可能日・スキルレベル・自己紹介文を特徴量として利用し、重み付きスコアリングにより推薦を行います。

---

## 背景

バンドメンバー募集では以下の課題があります。

- ジャンルや活動目的のミスマッチ
- 活動頻度やスケジュールの不一致
- 地域が離れていることによる活動困難

本プロジェクトではこれらの課題を解決するために、ユーザー間の相性を定量化し、推薦する仕組みを構築しました。

---

## 使用技術

- Python 3.11
- pandas
- scikit-learn
- FastAPI
- pytest

---

## 推薦ロジック

以下の特徴量を用いてスコアリングを行います。

| 特徴量       | 手法                  |
| ------------ | --------------------- |
| ジャンル     | Jaccard類似度         |
| 活動頻度     | 完全一致              |
| 活動目的     | 完全一致              |
| スキルレベル | 完全一致              |
| 活動可能日   | Jaccard類似度         |
| 自己紹介文   | TF-IDF + Cosine類似度 |

最終スコアは重み付きで計算します。

```text
score = 0.35 * genre
      + 0.20 * frequency
      + 0.20 * purpose
      + 0.10 * skill
      + 0.10 * available_days
      + 0.05 * bio
```

### 設計方針

- 地域は「類似度」ではなくフィルタとして扱う
- ユーザーが納得できるよう推薦理由を返す

---

## API仕様

### エンドポイント

```text
GET /users/{user_id}/recommendations
```

### クエリパラメータ

| パラメータ         | 型   | デフォルト | 説明                   |
| ------------------ | ---- | ---------- | ---------------------- |
| top_k              | int  | 5          | 取得件数（1〜20）      |
| same_location_only | bool | true       | 同一地域のみ推薦するか |

### レスポンス例

```json
[
  {
    "user_id": 2,
    "name": "GoodMatch",
    "age": 27,
    "location": "Osaka",
    "instrument": "bass",
    "genres": "rock,pop",
    "skill_level": "intermediate",
    "purpose": "cover_band",
    "frequency": "weekly",
    "available_days": "saturday",
    "score": 0.87,
    "reasons": [
      "共通ジャンル: rock, pop",
      "活動頻度が一致",
      "活動目的が一致",
      "スキルレベルが近い",
      "活動地域が一致"
    ]
  }
]
```

### ステータスコード

| ステータス | 説明                 |
| ---------- | -------------------- |
| 200        | 正常                 |
| 404        | ユーザーが存在しない |
| 422        | パラメータエラー     |

---

## セットアップ

```bash
cd backend
uv sync
```

---

## サーバー起動

```bash
uv run uvicorn app.main:app --reload
```

ブラウザで以下にアクセスします。

```text
http://127.0.0.1:8000/docs
```

---

## テスト

```bash
uv run pytest
```

---

## プロジェクト構成

```text
.
├── backend
│   ├── app
│   │   ├── main.py
│   │   ├── recommender.py
│   │   ├── schemas.py
│   │   └── data
│   ├── tests
│   └── pyproject.toml
├── notebooks
│   └── eda.ipynb
├── scripts
│   └── generate_dummy_users.py
└── docs
    └── portfolio.md
```

---

## 工夫点

- EDAに基づいた特徴量設計
- 重み付きスコアリングによる柔軟な推薦
- 推薦理由の可視化
- APIとして利用可能な構成
- pytestによるテスト実装

---

## 今後の改善

- 協調フィルタリングの導入
- ユーザー行動データの活用
- 学習ベースのランキングモデル（Learning to Rank）
- フロントエンドの実装
- Dockerによる本番環境構築
