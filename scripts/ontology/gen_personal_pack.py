# Generates kleos-core/src/ontology_personal.json (the default `personal` pack).
#   python3 scripts/ontology/gen_personal_pack.py kleos-core/src/ontology_personal.json
import json, sys
C = lambda name, parents, doc, aliases=(): {"name": name, "parents": list(parents), "label": name.lower(), "doc": doc, "aliases": list(aliases)}
classes = [
    C("Thing", [], "anything"),
    C("Person", ["Thing"], "a human being", ["people", "someone"]),
    C("Place", ["Thing"], "a city, country, address or venue", ["location", "city", "country"]),
    C("Organization", ["Thing"], "a company, school, team or other organisation", ["company", "employer", "school"]),
    C("Pet", ["Thing"], "an animal someone keeps", ["animal"]),
    C("Item", ["Thing"], "an object someone owns, uses or bought", ["object", "possession"]),
    C("Event", ["Thing"], "something that happens at a time"),
    C("Activity", ["Event"], "something someone does or did"),
    C("Media", ["Thing"], "a book, film, show, song or game"),
]
P = []
def prop(name, doc, kind="data", domain=("Person",), rng=(), card="one", temporal=True, datatype=None, unit=None,
         symmetric=False, transitive=False, inverse=None, parents=(), chain=(), aliases=(), cues=()):
    P.append({"name": name, "kind": kind, "domain": list(domain), "range": list(rng), "datatype": datatype, "card": card,
              "symmetric": symmetric, "transitive": transitive, "inverse": inverse, "parents": list(parents),
              "chain": list(chain), "temporal": temporal, "unit": unit, "label": name.replace("_", " "), "doc": doc,
              "aliases": list(aliases), "cues": list(cues)})
O = lambda *a, **k: prop(*a, kind="object", **k)
# where and who
O("lives_in", "where someone lives or moved to", rng=["Place"], aliases=["residence", "home city"],
  cues=["live in", "lives in", "living in", "moved to", "moving to", "relocated to", "my apartment in", "my flat in", "based in", "reside in"])
O("born_in", "where someone was born or grew up", rng=["Place"], temporal=False, aliases=["hometown"],
  cues=["born in", "grew up in", "my hometown", "originally from"])
prop("home", "the house or apartment someone lives in, buying a home, renovation", aliases=["house", "apartment"],
     cues=["my house", "my apartment", "my flat", "bought a house", "new house", "renovat*", "the house"])
O("employer", "the company or organisation someone works for", rng=["Organization"], aliases=["company"],
  cues=["work at", "work for", "working at", "working for", "works at", "works for", "joined", "hired", "my company", "my employer", "quit my job", "laid off"])
prop("job_title", "someone's job, role or position", aliases=["job", "role", "position", "occupation"],
     cues=["work as", "working as", "works as", "my job", "my role", "promoted", "position as", "job title"])
prop("income", "salary, pay, raise or income", datatype="xsd:decimal", unit="currency", aliases=["salary", "pay"],
     cues=["salary", "my pay", "a raise", "income", "i earn", "i make"])
prop("commute", "how someone gets to work and how long it takes", cues=["commute", "to work by", "drive to work"])
prop("education", "school, degree, course or class someone takes", aliases=["studies", "degree"],
     cues=["studying", "my degree", "a course", "my class", "classes", "university", "college", "graduated", "master's", "phd"])
# relations between people
O("partner", "someone's partner, spouse, boyfriend or girlfriend", rng=["Person"], symmetric=True, parents=["relative_of"],
  aliases=["spouse", "husband", "wife", "boyfriend", "girlfriend"],
  cues=["my partner", "my boyfriend", "my girlfriend", "my wife", "my husband", "my fiance", "my fiancee", "dating", "married", "engaged", "broke up"])
prop("relationship_status", "whether someone is single, dating, married or divorced", cues=["i'm single", "got married", "got divorced", "relationship status"])
O("sibling_of", "a brother or sister", rng=["Person"], card="many", temporal=False, symmetric=True, parents=["relative_of"], aliases=["sibling"],
  cues=["have a sister", "have a brother", "is my sister", "is my brother", "siblings", "have two sisters", "have two brothers"])
O("sister_of", "is the sister of", rng=["Person"], card="many", temporal=False, parents=["sibling_of"], aliases=["sister"])
O("brother_of", "is the brother of", rng=["Person"], card="many", temporal=False, parents=["sibling_of"], aliases=["brother"])
O("parent_of", "is a parent of", rng=["Person"], card="many", temporal=False, inverse="child_of", parents=["relative_of"], aliases=["parent"],
  cues=["have a son", "have a daughter", "my son", "my daughter", "my kids", "my children", "my baby"])
O("child_of", "is a child of", rng=["Person"], card="many", temporal=False, parents=["relative_of"], aliases=["child"], cues=["my mom", "my dad", "my mother", "my father", "my parents"])
O("mother_of", "is the mother of", rng=["Person"], card="many", temporal=False, parents=["parent_of"], aliases=["mother", "mom"])
O("father_of", "is the father of", rng=["Person"], card="many", temporal=False, parents=["parent_of"], aliases=["father", "dad"])
O("aunt_of", "is an aunt of (a sister of a parent)", rng=["Person"], card="many", temporal=False, chain=["sister_of", "parent_of"], parents=["relative_of"])
O("uncle_of", "is an uncle of (a brother of a parent)", rng=["Person"], card="many", temporal=False, chain=["brother_of", "parent_of"], parents=["relative_of"])
O("grandparent_of", "is a grandparent of", rng=["Person"], card="many", temporal=False, chain=["parent_of", "parent_of"], parents=["relative_of"], aliases=["grandparent"],
  cues=["my grandma", "my grandpa", "my grandmother", "my grandfather", "my grandparents"])
O("relative_of", "a family member: the super-property of the kinship relations (inferred, not typed from text)", rng=["Person"], card="many", temporal=False,
  symmetric=True, aliases=["relative", "relatives", "family member", "family members", "kin", "kinship"])
prop("children", "how many children someone has, pregnancy", cues=["kids", "children", "pregnant", "expecting a baby"])
prop("family", "parents, siblings and relatives and what they do", cues=["my family", "my relatives", "my cousin", "my aunt", "my uncle"])
O("friend_of", "a friend", rng=["Person"], card="many", symmetric=True, aliases=["friend"],
  cues=["my friend", "best friend", "friends with", "a friend of mine"])
O("boss_of", "is the manager or boss of", rng=["Person"], inverse="reports_to", aliases=["manager", "boss"], cues=["my boss", "my manager"])
O("reports_to", "reports to (has as boss)", rng=["Person"], cues=[])
# things owned and done
prop("pets", "pets someone has and how many", aliases=["pet", "dog", "cat"],
     cues=["my dog", "my cat", "my pet", "my puppy", "my kitten", "dogs", "cats", "pets", "adopted a"])
prop("vehicle", "car, bike or motorcycle someone owns or rides", aliases=["car", "bike"],
     cues=["my car", "drive a", "bought a car", "my bike", "motorcycle", "my tesla", "road bike"])
prop("device", "phone, computer, camera or other gadget", cues=["my phone", "my laptop", "my computer", "my camera", "iphone", "new phone"])
prop("collection", "items someone collects and how many", cues=["collect", "my collection", "postcards", "stamps", "vinyl", "records"])
prop("purchase", "something someone bought and the price", cues=["bought", "purchased", "paid", "spent"])
prop("home_loan", "mortgage, loan, pre-approval or debt amount", unit="currency", aliases=["mortgage", "loan"],
     cues=["mortgage", "loan", "pre-approved", "pre-approval", "debt", "down payment"])
prop("rent_or_bills", "rent, utilities and bills", unit="currency", aliases=["rent"], cues=["rent", "utilities", "bills", "electricity bill"])
prop("savings", "savings, budget, spending and investments", unit="currency", aliases=["budget", "investments"],
     cues=["savings", "saved", "budget", "invest*", "my portfolio", "emergency fund"])
prop("social_media", "followers, posts, accounts and online audience", datatype="xsd:integer",
     aliases=["follower count", "followers"], cues=["followers", "instagram", "youtube", "tiktok", "twitter", "subscribers", "my channel", "my blog"])
# habits and progress (single-valued state that changes)
prop("hobby", "a hobby or craft and how far someone has got in it", cues=["hobby", "pottery", "painting", "knitting", "woodworking", "sculpt*", "photography"])
prop("workout_routine", "gym, exercise and training frequency", cues=["gym", "work out", "workout", "exercise", "yoga", "training", "lift"])
prop("running", "runs, races, times and distances", cues=["run", "running", "5k", "10k", "marathon", "personal best", "race", "triathlon"])
prop("diet", "what someone eats or avoids, diet plan", cues=["vegan", "vegetarian", "diet", "i eat", "gluten", "keto", "calories"])
prop("health", "illness, condition, injury, allergy or medication", cues=["allergic", "allergy", "diagnosed", "injury", "injured", "medication", "doctor", "sick", "asthma"])
prop("body_measure", "weight, height or a fitness measure", cues=["weigh", "my weight", "lost", "pounds", "kg", "height", "bmi"])
prop("sleep", "sleep habits and hours", cues=["sleep", "bedtime", "wake up"])
prop("drink_habit", "coffee, tea or alcohol habits", cues=["coffee", "tea", "drink", "beer", "wine", "cocktail", "negroni", "espresso"])
prop("reading", "books someone reads or has read and how many", cues=["reading", "read", "books", "novel", "page"])
prop("watching", "shows, films and videos watched and how many", cues=["watching", "watched", "episodes", "series", "movie", "films", "videos"])
prop("music", "instrument, band, songs or concerts", cues=["guitar", "piano", "violin", "band", "concert", "playlist", "songs", "drums"])
prop("cooking", "recipes, cooking, baking and how many dishes", cues=["recipe", "cook", "bake", "baking", "dish", "chef"])
prop("garden", "plants and gardening", cues=["garden", "plants", "tomatoes", "seeds", "succulent"])
prop("volunteering", "volunteering, charity and donations", cues=["volunteer", "charity", "donat*", "fundrais*"])
prop("language", "learning a language and level", cues=["learning spanish", "learning french", "learning german", "duolingo", "fluent", "language"])
prop("project_progress", "progress on a project: pages, hours, percent done, count so far", cues=["so far", "hours", "finished", "completed", "progress", "chapters"])
prop("schedule", "when something regularly happens, days per week", cues=["every week", "times a week", "days a week", "every monday", "on saturdays", "weekly", "every morning"])
prop("membership", "subscriptions, memberships and plans", cues=["subscription", "membership", "member of", "subscribed", "plan"])
prop("favourite", "someone's favourite thing", cues=["favourite", "favorite"])
prop("age", "how old someone is", datatype="xsd:integer", cues=["years old", "turned", "my age"])
prop("birthday", "someone's birthday or date of birth", datatype="xsd:date", temporal=False, cues=["birthday", "born on"])
prop("contact", "phone number, e-mail or address", cues=["phone number", "email", "e-mail", "my address"])
# accumulate: never supersede
prop("likes", "something someone likes, loves or enjoys", card="many", cues=["i love", "i like", "i enjoy", "i'm into", "big fan of", "obsessed with"])
prop("dislikes", "something someone dislikes or hates", card="many", cues=["i hate", "i dislike", "can't stand", "not a fan of"])
O("visited", "a place someone went to or travelled to", rng=["Place"], card="many", temporal=False, aliases=["trip", "travel"],
  cues=["visited", "went to", "trip to", "travelled to", "traveled to", "vacation in", "holiday in", "flew to"])
prop("attended", "an event someone attended or will attend", card="many", temporal=False, aliases=["event"],
     cues=["attended", "went to a", "conference", "festival", "wedding", "party", "parade", "workshop", "meetup"])
prop("activity", "something someone did, is doing or will do at a point in time", card="many", temporal=False,
     cues=["went", "today i", "yesterday i", "last weekend", "this weekend", "i'm off to", "just got back", "took the kids", "camping", "hike", "swimming"])
prop("plan", "something someone plans, hopes or intends to do", card="many", cues=["planning to", "plan to", "thinking about", "going to", "hope to", "want to", "dream", "goal is", "next month", "someday"])
prop("opinion", "what someone thinks or feels about something", card="many", cues=["i think", "i feel", "i believe", "in my opinion", "it was", "it's been"])
O("same_as", "the same individual under another key (a name and a relation key)", domain=["Thing"], rng=["Thing"], card="many", temporal=False, symmetric=True)
O("located_in", "a place is inside another place", domain=["Place"], rng=["Place"], card="many", temporal=False, transitive=True)
pack = {"onto": "personal", "version": "1", "doc": "kleos default personal-memory ontology (docs/specs/ontology.md)",
        "classes": classes, "properties": P}
json.dump(pack, open(sys.argv[1], "w"), indent=1)
print(len(P), "properties", len(classes), "classes")
