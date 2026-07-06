df = clean_dataframe(df)

row = df[df["subject"].str.contains("Itinerary Change")].iloc[0]

print(row["body_text"])
print("----------------")
print(row["search_text"])