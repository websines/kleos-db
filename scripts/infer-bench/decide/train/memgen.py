"""Programmatic memory-style pairs for decider-nano fine-tuning (kleos authors, no external data).

Four relation classes over (subject, attribute, value) facts:
  update        same subject + attribute, new value (incl. numbers, dates, money, counts)
  contradiction same subject + attribute, negation or correction
  compatible    same subject, different attribute (or an additive detail of the same one)
  duplicate     paraphrase of the same fact
plus same/different-entity mention pairs and graph/ontology items (entity typing, relation
mapping among candidate relations). Everything is drawn from the word lists below with a
seeded RNG, so the output is reproducible.
"""
import random

FIRST = """Amara Bilal Carmen Dmitri Elif Farid Giulia Hamid Ingrid Jonas Kavya Lars Mei Nikolai Olu Paloma
Quentin Rania Stefan Tamsin Umar Valeria Wen Xavier Yara Zoltan Adaeze Bruno Chiara Darius Esme Femi Gwen
Haruto Isla Jae Kofi Lucia Marek Noor Oskar Pilar Radu Saanvi Teodor Uma Vikram Wanjiru Ximena Yusuf Zara
Anouk Bastian Cyrus Delphine Emeka Freya Gustavo Hana Ilya Jasmin Kenta Leila Matteo Nadira Orla Pavel
Rosa Samir Thandiwe Ulrich Vesna Wilhelmina Yannick Zofia Arjun Beatriz Callum Dalia Enzo Farah Goran""".split()
LAST = """Okafor Lindqvist Moreau Petrov Haddad Kowalski Nakamura Sato Fernandes Brennan Varga Osei Dubois
Castillo Novak Mensah Rahman Jansen Ferreira Kaur Ivanova Takahashi Romano Olsen Achebe Silva Horvat
Bergstrom Delgado Nwosu Rossi Abiodun Lemaire Kovac Andersen Mbeki Quispe Yilmaz Duarte Hoffmann""".split()
CITIES = """Porto Montreal Nairobi Kraków Osaka Valencia Gothenburg Adelaide Bogotá Ljubljana Hanoi Leeds
Seattle Austin Dublin Munich Cape_Town Edinburgh Lyon Taipei Accra Lima Riga Brno Busan Perth Halifax
Bordeaux Tbilisi Utrecht Chennai Denver Glasgow Bilbao Malmö Wellington Antwerp Guadalajara Kyoto""".split()
CITIES = [c.replace("_", " ") for c in CITIES]
COMPANIES = """Shopify Atlassian Spotify Canva Datadog Zalando Monzo Klarna Figma Notion Airbnb Siemens
Unilever Ocado Rakuten Wise Revolut Grafana Elastic Twilio HubSpot Mozilla Cloudflare Bosch Philips
Novartis Maersk Ericsson Nokia Adyen Booking.com Deliveroo Etsy Asana Miro Sennheiser""".split()
JOBS = ["nurse", "civil engineer", "pharmacist", "high school teacher", "UX researcher", "paralegal",
        "sous chef", "data engineer", "physiotherapist", "architect", "product manager", "translator",
        "electrician", "veterinarian", "copywriter", "airline pilot", "dental hygienist", "accountant",
        "site reliability engineer", "museum curator", "social worker", "barista", "graphic designer"]
SPORTS = ["badminton", "rowing", "bouldering", "squash", "volleyball", "fencing", "cricket", "surfing",
          "table tennis", "cycling", "judo", "golf", "basketball", "kayaking", "netball"]
FOODS = ["ramen", "paella", "falafel", "pho", "lasagna", "biryani", "tacos", "sushi", "pierogi", "dumplings",
         "shakshuka", "jollof rice", "moussaka", "pad thai", "risotto"]
COLORS = ["teal", "maroon", "mustard yellow", "navy blue", "forest green", "lavender", "charcoal grey",
          "burnt orange", "white", "silver", "black"]
LANGS = ["Rust", "Python", "Kotlin", "TypeScript", "Elixir", "Haskell", "C#", "Swift", "Scala", "Julia", "Ruby"]
EDITORS = ["Emacs", "Sublime Text", "IntelliJ IDEA", "Zed", "Helix", "Vim", "PyCharm", "Xcode"]
INSTR = ["cello", "saxophone", "ukulele", "drums", "violin", "piano", "banjo", "flute", "harp"]
PETS = [("dog", "Biscuit"), ("cat", "Pepper"), ("parrot", "Kiwi"), ("rabbit", "Clover"), ("dog", "Maple"),
        ("cat", "Miso"), ("tortoise", "Sheldon"), ("hamster", "Nugget"), ("dog", "Juniper"), ("cat", "Tofu")]
CARS = ["Toyota Corolla", "Volkswagen Golf", "Tesla Model 3", "Honda Civic", "Subaru Outback", "Kia Niro",
        "Ford Fiesta", "Mazda CX-5", "Volvo XC40", "Skoda Octavia", "Renault Clio"]
DIETS = ["vegetarian", "vegan", "pescatarian", "gluten-free", "keto"]
UNIS = ["the University of Porto", "McGill", "ETH Zurich", "the University of Nairobi", "KU Leuven",
        "Trinity College Dublin", "the University of Melbourne", "TU Delft", "Seoul National University"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
HOSTS = ["AWS", "Google Cloud", "Azure", "Hetzner", "DigitalOcean", "Render", "Vercel", "Netlify", "Linode"]
OSES = ["Ubuntu 22.04", "Debian 11", "Fedora 39", "Alpine 3.18", "Rocky Linux 9", "Windows Server 2019",
        "FreeBSD 13", "Ubuntu 24.04", "Debian 12"]
DBS = ["PostgreSQL 13", "PostgreSQL 16", "MySQL 8", "MongoDB 6", "SQLite", "Redis 7", "CockroachDB", "MariaDB 10"]
TEAMS = ["the payments team", "the search team", "the mobile team", "the infra team", "the data platform team",
         "the growth team", "the security team", "the billing team"]
DOCTORS = ["Dr. Varga", "Dr. Osei", "Dr. Lemaire", "Dr. Kaur", "Dr. Brennan", "Dr. Horvat", "Dr. Yilmaz"]
ALLERGIES = ["peanuts", "shellfish", "penicillin", "pollen", "cats", "latex", "sesame"]
HOBBIES = ["woodworking", "birdwatching", "knitting", "salsa dancing", "rock climbing", "baking bread",
           "chess", "gardening", "photography", "pottery", "sailing", "calligraphy"]
GYMS = ["PureGym", "Anytime Fitness", "Gold's Gym", "a local CrossFit box", "the YMCA", "Equinox"]
COMMUTE = [("cycles", "by bike"), ("takes the bus", "by bus"), ("drives", "by car"), ("walks", "on foot"),
           ("takes the tram", "by tram"), ("takes the metro", "by metro")]
PROJECTS = ["Project Atlas", "the Falcon rewrite", "the Orion dashboard", "Project Lumen", "the Kestrel API",
            "the Nimbus migration", "Project Harbor", "the Quartz app"]
ORGS = ["Acme Robotics", "Northwind Labs", "Bluefin Analytics", "Helios Energy", "Copperleaf Studio",
        "Granite Logistics", "Tidewater Health", "Pinecrest Bank", "Silverline Media", "Juniper Foods"]


def money(rng, lo, hi, step=50, cur=None):
    cur = cur or rng.choice(["$", "€", "£"])
    v = rng.randrange(lo, hi, step)
    return v, f"{cur}{v:,}"


def date(rng):
    return f"{rng.choice(MONTHS)} {rng.randint(1, 28)}"


def year(rng, lo=1950, hi=2004):
    return str(rng.randint(lo, hi))


def different(rng, pool, old):
    for _ in range(50):
        v = rng.choice(pool)
        if v != old:
            return v
    raise ValueError("pool too small")


def person(rng):
    if rng.random() < 0.18:
        return "the user", "The user", "the user's", "The user's", "they"
    n = rng.choice(FIRST)
    return n, n, f"{n}'s", f"{n}'s", None


# --- attribute schemas -------------------------------------------------------------------------
# Each returns a dict with base / dup / update / contra phrasings for one subject. `S` is the
# subject as sentence start, `s` mid-sentence, `P` / `p` possessive.

def a_city(rng, S, s, P, p):
    old = rng.choice(CITIES); new = different(rng, CITIES, old)
    return dict(
        base=[f"{S} lives in {old}.", f"{P} home is in {old}.", f"{S} is based in {old}."],
        dup=[f"{P} home city is {old}.", f"{S} resides in {old}.", f"{old} is where {s} lives."],
        upd=[f"{S} moved to {new} {rng.choice(['last month', 'in the summer', 'two weeks ago', 'this year'])}.",
             f"{S} relocated from {old} to {new}.", f"{S} now lives in {new} after leaving {old}.",
             f"{S} has settled in {new} since the move."],
        con=[f"{S} has never lived in {old}.", f"{S} does not live in {old}; that was a misunderstanding.",
             f"Correction: {S} lives in {new}, not {old}."],
        attr="city", vals=(old, new))


def a_employer(rng, S, s, P, p):
    old = rng.choice(COMPANIES); new = different(rng, COMPANIES, old)
    return dict(
        base=[f"{S} works at {old}.", f"{S} is employed by {old}.", f"{P} employer is {old}."],
        dup=[f"{S} has a job at {old}.", f"{old} is where {s} works.", f"{S} is on the payroll at {old}."],
        upd=[f"{S} left {old} and joined {new}.", f"{S} started a new job at {new} last week.",
             f"{S} now works for {new}.", f"{S} quit {old}; {s} accepted an offer from {new}."],
        con=[f"{S} has never worked at {old}.", f"{S} does not work at {old}, {s} only interviewed there.",
             f"Correction: {P} employer is {new}, not {old}."],
        attr="employer", vals=(old, new))


def a_job(rng, S, s, P, p):
    old = rng.choice(JOBS); new = different(rng, JOBS, old)
    an = lambda w: ("an " if w[0] in "aeiou" else "a ") + w
    return dict(
        base=[f"{S} is {an(old)}.", f"{S} works as {an(old)}.", f"{P} job is {old}."],
        dup=[f"{S} earns a living as {an(old)}.", f"By profession, {s} is {an(old)}.", f"{P} occupation is {old}."],
        upd=[f"{S} retrained and is now {an(new)}.", f"{S} switched careers to become {an(new)}.",
             f"{S} stopped working as {an(old)} and now works as {an(new)}."],
        con=[f"{S} is not {an(old)}.", f"{S} has never worked as {an(old)}.",
             f"Correction: {s} is {an(new)}, not {an(old)}."],
        attr="job", vals=(old, new))


def a_partner(rng, S, s, P, p):
    old = rng.choice(FIRST); new = different(rng, FIRST, old)
    return dict(
        base=[f"{S} is dating {old}.", f"{P} partner is {old}.", f"{S} is in a relationship with {old}."],
        dup=[f"{old} is {p} partner.", f"{S} and {old} are a couple.", f"{S} is seeing {old}."],
        upd=[f"{S} and {old} split up last spring.", f"{S} broke up with {old} and is now seeing {new}.",
             f"{S} and {old} ended their relationship."],
        con=[f"{S} has never dated {old}.", f"{S} and {old} are just friends, they were never a couple."],
        attr="partner", vals=(old, new))


def a_pet(rng, S, s, P, p):
    (kind, name) = rng.choice(PETS)
    (kind2, name2) = different(rng, PETS, (kind, name))
    return dict(
        base=[f"{S} has a {kind} named {name}.", f"{S} owns a {kind} called {name}."],
        dup=[f"{P} {kind} is called {name}.", f"{name} is {p} {kind}."],
        upd=[f"{name}, {p} {kind}, died last month, and {s} has no pets now.",
             f"{S} rehomed {name} and adopted a {kind2} named {name2} instead.",
             f"{S} gave {name} to a relative and no longer has a {kind}."],
        con=[f"{S} has never owned a {kind}.", f"{name} is not {p} {kind}; it belongs to a neighbour."],
        attr="pet", vals=(name, name2))


def a_fav(rng, S, s, P, p):
    cat, pool = rng.choice([("sport", SPORTS), ("food", FOODS), ("colour", COLORS), ("instrument", INSTR),
                            ("programming language", LANGS), ("editor", EDITORS)])
    old = rng.choice(pool); new = different(rng, pool, old)
    return dict(
        base=[f"{P} favourite {cat} is {old}.", f"{S} likes {old} best of any {cat}."],
        dup=[f"{old.capitalize() if old[0].islower() else old} is {p} favourite {cat}.",
             f"Of all {cat}s, {s} prefers {old}."],
        upd=[f"These days {P.lower() if P[0] == 'T' else P} favourite {cat} is {new}, not {old}.",
             f"{S} has gone off {old}; {p} favourite {cat} is now {new}.",
             f"{S} says {new} has replaced {old} as {p} favourite {cat}."],
        con=[f"{S} actually dislikes {old}.", f"{old.capitalize() if old[0].islower() else old} is not {p} favourite {cat} at all."],
        attr=f"favourite {cat}", vals=(old, new))


def a_age(rng, S, s, P, p):
    old = rng.randint(19, 70); new = old + 1
    wrong = old + rng.choice([-7, -3, 4, 6, 10])
    return dict(
        base=[f"{S} is {old} years old.", f"{S} is {old}."],
        dup=[f"{P} age is {old}.", f"{S} is aged {old}."],
        upd=[f"{S} turned {new} last week.", f"{S} just celebrated {p} {new}th birthday."],
        con=[f"{S} is actually {wrong}, not {old}.", f"Correction: {s} is {wrong} years old."],
        attr="age", vals=(str(old), str(new)))


def a_salary(rng, S, s, P, p):
    v, old = money(rng, 38000, 160000, 1000)
    cur = old[0]
    nv = v + rng.choice([3000, 5000, 8000, 12000, -4000])
    new = f"{cur}{nv:,}"
    return dict(
        base=[f"{P} salary is {old} a year.", f"{S} earns {old} per year."],
        dup=[f"{S} makes {old} annually.", f"{P} annual pay is {old}."],
        upd=[f"After the review, {p} salary went up to {new}." if nv > v else f"{P} salary was cut to {new}.",
             f"{S} now earns {new} a year.", f"{P} pay changed to {new} per year this quarter."],
        con=[f"{S} has never earned {old}; the figure was wrong.", f"Correction: {p} salary is {new}, not {old}."],
        attr="salary", vals=(old, new))


def a_rent(rng, S, s, P, p):
    v, old = money(rng, 600, 3500, 50)
    cur = old[0]; nv = v + rng.choice([50, 100, 150, 250, -100]); new = f"{cur}{nv:,}"
    return dict(
        base=[f"{P} rent is {old} per month.", f"{S} pays {old} a month in rent."],
        dup=[f"Monthly rent for {p} flat is {old}.", f"{S} spends {old} monthly on rent."],
        upd=[f"The landlord raised {p} rent to {new} a month." if nv > v else f"{P} rent dropped to {new} a month.",
             f"{S} now pays {new} per month in rent.", f"From next month {p} rent is {new}."],
        con=[f"Correction: {p} rent is {new} per month, not {old}.", f"{S} does not pay {old} in rent."],
        attr="rent", vals=(old, new))


def a_kids(rng, S, s, P, p):
    w = ["one", "two", "three", "four"]
    i = rng.randint(0, 2)
    old, new = w[i], w[i + 1]
    child = "child" if old == "one" else "children"
    return dict(
        base=[f"{S} has {old} {child}.", f"{S} is a parent of {old} {child}."],
        dup=[f"{P} family includes {old} {child}.", f"{S} is raising {old} {child}."],
        upd=[f"{S} had another baby, so {s} now has {new} children.", f"{S} now has {new} children after the adoption."],
        con=[f"{S} has no children.", f"{S} actually has {w[(i + 2) % 4]} children, not {old}."],
        attr="children", vals=(old, new))


def a_birthday(rng, S, s, P, p):
    m = rng.choice(MONTHS); d = rng.randint(1, 28); d2 = d + rng.choice([1, 2, 9, -1]) if d > 2 else d + 3
    old, wrong = f"{m} {d}", f"{m} {d2}"
    return dict(
        base=[f"{P} birthday is on {old}.", f"{S} was born on {old}."],
        dup=[f"{S} celebrates {p} birthday on {old}.", f"{P} birthday falls on {old}."],
        upd=[f"Correction: {p} birthday is {wrong}; I mixed up the date.", f"{P} birthday is actually on {wrong}."],
        con=[f"{P} birthday is not in {m}.", f"{S} was not born on {old}."],
        attr="birthday", vals=(old, wrong), upd_is_contra=True)


def a_car(rng, S, s, P, p):
    old = rng.choice(CARS); new = different(rng, CARS, old)
    return dict(
        base=[f"{S} drives a {old}.", f"{P} car is a {old}."],
        dup=[f"{S} owns a {old}.", f"The car {s} drives is a {old}."],
        upd=[f"{S} sold the {old} and bought a {new}.", f"{S} traded in {p} {old} for a {new}.",
             f"{S} no longer has a car since selling the {old}."],
        con=[f"{S} has never owned a {old}.", f"{S} does not drive; {s} has no car."],
        attr="car", vals=(old, new))


def a_diet(rng, S, s, P, p):
    old = rng.choice(DIETS); new = different(rng, DIETS, old)
    return dict(
        base=[f"{S} is {old}.", f"{S} follows a {old} diet."],
        dup=[f"{S} eats a {old} diet.", f"{P} diet is {old}."],
        upd=[f"{S} stopped being {old} and now eats {new}.", f"{S} switched from a {old} to a {new} diet this year."],
        con=[f"{S} is not {old}; {s} eats everything.", f"{S} has never been {old}."],
        attr="diet", vals=(old, new))


def a_languages(rng, S, s, P, p):
    w = ["two", "three", "four", "five"]
    i = rng.randint(0, 2); old = w[i]; new = w[i + 1]
    return dict(
        base=[f"{S} speaks {old} languages.", f"{S} is fluent in {old} languages."],
        dup=[f"{S} can speak {old} languages.", f"The number of languages {s} speaks is {old}."],
        upd=[f"{S} has learned another language and now speaks {new}.", f"{S} now speaks {new} languages after a year abroad."],
        con=[f"{S} only speaks one language.", f"{S} actually speaks {w[(i + 2) % 4]} languages, not {old}."],
        attr="languages", vals=(old, new))


def a_school(rng, S, s, P, p):
    old = rng.choice(UNIS); new = different(rng, UNIS, old)
    return dict(
        base=[f"{S} studies at {old}.", f"{S} is a student at {old}."],
        dup=[f"{S} is enrolled at {old}.", f"{old} is where {s} studies."],
        upd=[f"{S} transferred from {old} to {new}.", f"{S} is now studying at {new}."],
        con=[f"{S} never studied at {old}.", f"{S} is not a student at {old}; {s} only applied."],
        attr="university", vals=(old, new))


def a_hobby(rng, S, s, P, p):
    old = rng.choice(HOBBIES); new = different(rng, HOBBIES, old)
    return dict(
        base=[f"{S} spends weekends on {old}.", f"{P} main hobby is {old}."],
        dup=[f"{S} does {old} in {p} free time.", f"{old.capitalize()} is {p} favourite pastime."],
        upd=[f"{S} gave up {old} and took up {new}.", f"{S} stopped {old}; these days it is all {new}."],
        con=[f"{S} has never tried {old}.", f"{S} actually hates {old}."],
        attr="hobby", vals=(old, new))


def a_commute(rng, S, s, P, p):
    (v1, m1) = rng.choice(COMMUTE); (v2, m2) = different(rng, COMMUTE, (v1, m1))
    return dict(
        base=[f"{S} {v1} to work.", f"{S} commutes {m1}."],
        dup=[f"{S} gets to work {m1}.", f"{P} commute is {m1}."],
        upd=[f"{S} now {v2} to work instead.", f"{S} stopped commuting {m1} and now goes {m2}."],
        con=[f"{S} never goes to work {m1}.", f"{S} works from home and does not commute at all."],
        attr="commute", vals=(m1, m2))


def a_doctor(rng, S, s, P, p):
    kind = rng.choice(["dentist", "GP", "physiotherapist", "dermatologist"])
    old = rng.choice(DOCTORS); new = different(rng, DOCTORS, old)
    return dict(
        base=[f"{P} {kind} is {old}.", f"{S} sees {old} as {p} {kind}."],
        dup=[f"{old} is {p} {kind}.", f"{S} goes to {old} for {kind} visits."],
        upd=[f"{S} switched {kind}s from {old} to {new}.", f"{P} new {kind} is {new}."],
        con=[f"{old} has never been {p} {kind}.", f"Correction: {p} {kind} is {new}, not {old}."],
        attr=kind, vals=(old, new))


def a_allergy(rng, S, s, P, p):
    old = rng.choice(ALLERGIES); new = different(rng, ALLERGIES, old)
    return dict(
        base=[f"{S} is allergic to {old}.", f"{S} has an allergy to {old}."],
        dup=[f"{old.capitalize()} give {s} an allergic reaction." if old.endswith("s") else f"{old.capitalize()} gives {s} an allergic reaction.",
             f"{S} cannot have {old} because of an allergy."],
        upd=[f"A new test showed {s} is no longer allergic to {old}.", f"{S} outgrew {p} allergy to {old}."],
        con=[f"{S} is not allergic to {old}.", f"{S} has no allergies at all."],
        attr="allergy", vals=(old, new))


def a_gym(rng, S, s, P, p):
    old = rng.choice(GYMS); new = different(rng, GYMS, old)
    return dict(
        base=[f"{S} has a membership at {old}.", f"{S} trains at {old}."],
        dup=[f"{P} gym is {old}.", f"{S} works out at {old}."],
        upd=[f"{S} cancelled {old} and signed up at {new}.", f"{S} now trains at {new} instead."],
        con=[f"{S} has never been a member of {old}.", f"{S} does not go to {old}."],
        attr="gym", vals=(old, new))


PERSON_ATTRS = [a_city, a_employer, a_job, a_partner, a_pet, a_fav, a_age, a_salary, a_rent, a_kids, a_birthday,
                a_car, a_diet, a_languages, a_school, a_hobby, a_commute, a_doctor, a_allergy, a_gym]

# --- organisation / system / project subjects ------------------------------------------------


def o_hq(rng, S):
    old = rng.choice(CITIES); new = different(rng, CITIES, old)
    return dict(base=[f"{S} is headquartered in {old}.", f"{S}'s head office is in {old}."],
                dup=[f"{S} has its headquarters in {old}.", f"The main office of {S} is in {old}."],
                upd=[f"{S} moved its headquarters from {old} to {new}.", f"{S}'s head office is now in {new}."],
                con=[f"{S} has never had an office in {old}.", f"Correction: {S} is headquartered in {new}, not {old}."],
                attr="hq", vals=(old, new))


def o_ceo(rng, S):
    old = f"{rng.choice(FIRST)} {rng.choice(LAST)}"; new = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    return dict(base=[f"{old} is the CEO of {S}.", f"{S} is run by CEO {old}."],
                dup=[f"The chief executive of {S} is {old}.", f"{S}'s CEO is {old}."],
                upd=[f"{old} stepped down and {new} became CEO of {S}.", f"{S} appointed {new} as its new CEO."],
                con=[f"{old} has never been CEO of {S}.", f"Correction: the CEO of {S} is {new}, not {old}."],
                attr="ceo", vals=(old, new))


def o_headcount(rng, S):
    old = rng.randrange(20, 4000, 10); new = old + rng.choice([-120, -40, 35, 80, 250])
    return dict(base=[f"{S} has {old:,} employees.", f"{S} employs {old:,} people."],
                dup=[f"The headcount at {S} is {old:,}.", f"{old:,} people work at {S}."],
                upd=[f"After the latest round of hiring, {S} has {new:,} employees." if new > old else f"After layoffs, {S} is down to {new:,} employees.",
                     f"{S} now employs {new:,} people."],
                con=[f"{S} has never had {old:,} employees; the real number is {new:,}.", f"Correction: {S} employs {new:,} people, not {old:,}."],
                attr="headcount", vals=(str(old), str(new)))


def o_price(rng, S):
    plan = rng.choice(["Pro plan", "basic subscription", "team plan", "annual licence", "starter tier"])
    v, old = money(rng, 5, 400, 1); cur = old[0]; nv = v + rng.choice([1, 2, 5, 10, -3]); new = f"{cur}{nv:,}"
    return dict(base=[f"The {plan} of {S} costs {old} a month.", f"{S} charges {old} per month for its {plan}."],
                dup=[f"{S}'s {plan} is priced at {old} monthly.", f"A month of the {plan} from {S} is {old}."],
                upd=[f"{S} raised the price of its {plan} to {new} a month." if nv > v else f"{S} cut the {plan} price to {new}.",
                     f"The {plan} of {S} now costs {new} per month."],
                con=[f"Correction: the {plan} of {S} costs {new}, not {old}.", f"The {plan} of {S} is free; it never cost {old}."],
                attr="price", vals=(old, new))


def o_release(rng, S):
    old, new = date(rng), date(rng)
    while new == old:
        new = date(rng)
    thing = rng.choice(["launch", "release", "go-live", "beta", "migration cutover"])
    return dict(base=[f"The {thing} of {S} is scheduled for {old}.", f"{S} plans its {thing} on {old}."],
                dup=[f"The {thing} date for {S} is {old}.", f"{S} will have its {thing} on {old}."],
                upd=[f"The {thing} of {S} has been pushed to {new}.", f"{S} rescheduled its {thing} to {new}.",
                     f"The {thing} of {S} moved from {old} to {new}."],
                con=[f"The {thing} of {S} was never planned for {old}.", f"Correction: the {thing} of {S} is on {new}, not {old}."],
                attr="release date", vals=(old, new))


def o_meeting(rng, S):
    old = rng.choice(DAYS); new = different(rng, DAYS, old)
    mt = rng.choice(["weekly sync", "stand-up review", "planning meeting", "retro", "board meeting"])
    return dict(base=[f"The {mt} of {S} is on {old}s.", f"{S} holds its {mt} every {old}."],
                dup=[f"Every {old}, {S} has its {mt}.", f"{S} meets on {old}s for the {mt}."],
                upd=[f"The {mt} of {S} moved to {new}s starting this month.", f"{S} now holds its {mt} on {new}s."],
                con=[f"{S} has never held its {mt} on a {old}.", f"Correction: the {mt} of {S} is on {new}s."],
                attr="meeting day", vals=(old, new))


def o_host(rng, S):
    old = rng.choice(HOSTS); new = different(rng, HOSTS, old)
    return dict(base=[f"{S} is hosted on {old}.", f"{S} runs on {old}."],
                dup=[f"{S} is deployed to {old}.", f"The hosting provider for {S} is {old}."],
                upd=[f"{S} migrated from {old} to {new}.", f"{S} now runs on {new} after leaving {old}."],
                con=[f"{S} has never run on {old}.", f"Correction: {S} is hosted on {new}, not {old}."],
                attr="hosting", vals=(old, new))


def o_os(rng, S):
    old = rng.choice(OSES); new = different(rng, OSES, old)
    return dict(base=[f"{S} runs {old}.", f"The operating system on {S} is {old}."],
                dup=[f"{S} is running {old}.", f"{old} is installed on {S}."],
                upd=[f"{S} was upgraded from {old} to {new}.", f"{S} now runs {new}."],
                con=[f"{S} is not running {old}.", f"Correction: {S} runs {new}, not {old}."],
                attr="os", vals=(old, new))


def o_db(rng, S):
    old = rng.choice(DBS); new = different(rng, DBS, old)
    return dict(base=[f"{S} stores its data in {old}.", f"The database behind {S} is {old}."],
                dup=[f"{S} uses {old} as its database.", f"{old} is the datastore for {S}."],
                upd=[f"{S} moved its data from {old} to {new}.", f"{S} now uses {new} as its database."],
                con=[f"{S} has never used {old}.", f"Correction: {S} uses {new}, not {old}."],
                attr="database", vals=(old, new))


def o_ram(rng, S):
    old = rng.choice([8, 16, 32, 64, 128]); new = old * 2
    return dict(base=[f"{S} has {old} GB of RAM.", f"{S} is provisioned with {old} GB of memory."],
                dup=[f"The memory on {S} is {old} GB.", f"{S} comes with {old} GB of RAM."],
                upd=[f"{S} was upgraded to {new} GB of RAM.", f"{S} now has {new} GB of memory."],
                con=[f"Correction: {S} has {new} GB of RAM, not {old}.", f"{S} does not have {old} GB of RAM."],
                attr="ram", vals=(f"{old} GB", f"{new} GB"))


def o_budget(rng, S):
    v, old = money(rng, 10000, 900000, 5000); cur = old[0]; nv = v + rng.choice([-20000, 15000, 40000, 100000]); new = f"{cur}{nv:,}"
    return dict(base=[f"The budget for {S} is {old}.", f"{S} has a budget of {old}."],
                dup=[f"{S} is funded with {old}.", f"{old} is the budget allocated to {S}."],
                upd=[f"The budget for {S} was increased to {new}." if nv > v else f"The budget for {S} was cut to {new}.",
                     f"{S} now has a budget of {new}."],
                con=[f"Correction: the budget for {S} is {new}, not {old}.", f"{S} never had a budget of {old}."],
                attr="budget", vals=(old, new))


def o_owner(rng, S):
    old = rng.choice(TEAMS); new = different(rng, TEAMS, old)
    return dict(base=[f"{S} is owned by {old}.", f"{old.capitalize()} maintains {S}."],
                dup=[f"{S} is maintained by {old}.", f"{old.capitalize()} is responsible for {S}."],
                upd=[f"Ownership of {S} moved from {old} to {new}.", f"{new.capitalize()} took over {S} this quarter."],
                con=[f"{old.capitalize()} has never owned {S}.", f"Correction: {S} belongs to {new}, not {old}."],
                attr="owner", vals=(old, new))


ORG_ATTRS = [o_hq, o_ceo, o_headcount, o_price, o_budget, o_meeting]
SYS_ATTRS = [o_host, o_os, o_db, o_ram, o_owner]
PROJ_ATTRS = [o_release, o_budget, o_owner, o_meeting]


def org_subject(rng):
    return rng.choice(ORGS)


def sys_subject(rng):
    return rng.choice(["the build server", "the staging cluster", "the main database host", "the web server",
                       "the CI runner", "the analytics box", "the API gateway", "the backup server"])


def proj_subject(rng):
    return rng.choice(PROJECTS)


def facts_for(rng):
    """One subject with all its attribute schemas, as (kind, subject, [schema dict])."""
    r = rng.random()
    if r < 0.62:
        n, S, p_, P, _ = person(rng)
        s = n if n != "the user" else "the user"
        p = p_
        fns = rng.sample(PERSON_ATTRS, 4)
        return "person", S, [f(rng, S, s, P, p) for f in fns]
    if r < 0.76:
        S = org_subject(rng); fns = rng.sample(ORG_ATTRS, 4)
    elif r < 0.9:
        S = sys_subject(rng); fns = rng.sample(SYS_ATTRS, 4)
    else:
        S = proj_subject(rng); fns = rng.sample(PROJ_ATTRS, 3)
    return "thing", S, [f(rng, S) for f in fns]


def fix_case(t):
    t = t.strip()
    return t[0].upper() + t[1:] if t else t


def memory_pairs(rng, n_subjects):
    """-> list of (old, new, cls, group). One subject gives one pair per class (when possible)."""
    out = []
    for g in range(n_subjects):
        kind, S, schemas = facts_for(rng)
        main, other = schemas[0], schemas[1:]
        old = rng.choice(main["base"])
        grp = f"mem{g}"
        upd = rng.choice(main["upd"])
        out.append((old, upd, "contradiction" if main.get("upd_is_contra") else "update", grp))
        out.append((old, rng.choice(main["con"]), "contradiction", grp))
        out.append((old, rng.choice(main["dup"]), "duplicate", grp))
        out.append((old, rng.choice(rng.choice(other)["base"]), "compatible", grp))
        if rng.random() < 0.5:  # a second, harder compatible: another attribute stated as a change
            o2 = rng.choice(other)
            out.append((old, rng.choice(o2["upd"]), "compatible", grp))
        if rng.random() < 0.35:  # duplicate in the other base phrasing
            alt = [b for b in main["base"] if b != old]
            if alt:
                out.append((old, rng.choice(alt), "duplicate", grp))
    return [(fix_case(a), fix_case(b), c, g) for a, b, c, g in out]


# --- entity mention pairs -----------------------------------------------------------------------
NICK = {"Robert": "Bob", "William": "Bill", "Katherine": "Kate", "Elizabeth": "Liz", "Alexander": "Alex",
        "Jonathan": "Jon", "Margaret": "Maggie", "Michael": "Mike", "Daniel": "Dan", "Samantha": "Sam",
        "Christopher": "Chris", "Rebecca": "Becky", "Nicholas": "Nick", "Victoria": "Vicky", "Benjamin": "Ben",
        "Theodore": "Teddy", "Abigail": "Abby", "Patricia": "Patty", "Frederick": "Fred", "Josephine": "Jo"}
ABBR = [("New York City", "NYC"), ("Los Angeles", "LA"), ("United Kingdom", "UK"),
        ("Massachusetts Institute of Technology", "MIT"), ("World Health Organization", "WHO"),
        ("San Francisco", "SF"), ("European Central Bank", "ECB"), ("International Monetary Fund", "IMF"),
        ("Rio de Janeiro", "Rio"), ("Amazon Web Services", "AWS"), ("Kubernetes", "k8s"),
        ("Saint Petersburg", "St. Petersburg"), ("Washington, D.C.", "DC")]
ROLES = ["the team lead", "the landlord", "the new hire", "the contractor", "the accountant", "the coach",
         "the neighbour", "the tutor", "the recruiter", "the vet"]


def entity_pairs(rng, n):
    out = []
    for g in range(n):
        grp = f"ent{g}"
        r = rng.random()
        if r < 0.3:
            full, nick = rng.choice(list(NICK.items())); last = rng.choice(LAST); co = rng.choice(COMPANIES)
            role = rng.choice(JOBS)
            a = f"{full} {last}, a {role} at {co}, sent the report."
            b = f"{nick} {last} from {co} followed up on the report."
            out.append((a, b, "same", grp))
            last2 = different(rng, LAST, last); co2 = different(rng, COMPANIES, co)
            out.append((a, f"{nick} {last2}, who works at {co2}, sent an invoice.", "different", grp))
        elif r < 0.5:
            longf, short = rng.choice(ABBR)
            ctx = rng.choice(["The workshop takes place in", "The user grew up in", "Our partner is", "The data came from"])
            out.append((f"{ctx} {longf}.", f"{short} is mentioned again in the follow-up email.", "same", grp))
            l2, s2 = different(rng, ABBR, (longf, short))
            out.append((f"{ctx} {longf}.", f"{s2} is mentioned again in the follow-up email.", "different", grp))
        elif r < 0.75:
            f = rng.choice(FIRST); l = rng.choice(LAST); city = rng.choice(CITIES); role = rng.choice(ROLES)
            a = f"{f} {l} is {role} who lives in {city}."
            out.append((a, f"{role.capitalize()}, {f}, mentioned {city} again.", "same", grp))
            city2 = different(rng, CITIES, city); l2 = different(rng, LAST, l)
            out.append((a, f"{f} {l2}, a retired {rng.choice(JOBS)} from {city2}, joined the club.", "different", grp))
        else:
            first = rng.choice(FIRST); rel = rng.choice(["sister", "brother", "cousin", "manager", "flatmate", "doctor"])
            a = f"The user's {rel} {first} is visiting next week."
            out.append((a, f"{first}, the user's {rel}, booked a train ticket.", "same", grp))
            rel2 = different(rng, ["sister", "brother", "cousin", "manager", "flatmate", "doctor"], rel)
            out.append((a, f"The user's {rel2} {first} {rng.choice(LAST)} is a different person who moved away.", "different", grp))
    return out


# --- graph / ontology items -----------------------------------------------------------------------
TYPES = {
    "person": "person: a human individual",
    "organization": "organization: a company, institution or team",
    "location": "location: a city, country, address or place",
    "event": "event: something that happens at a time",
    "product": "product: software, a device or a service",
    "date": "date: a calendar date or time",
}
RELS = {
    "works_at": "works_at: a person is employed by an organization",
    "lives_in": "lives_in: a person resides in a place",
    "married_to": "married_to: two people are spouses",
    "owns": "owns: someone owns an object or animal",
    "located_in": "located_in: a place or organization is inside another place",
    "scheduled_for": "scheduled_for: an event is planned for a date",
    "uses": "uses: a person or system uses a tool or product",
    "parent_of": "parent_of: someone is the parent of a person",
}
REL_FACTS = [
    ("works_at", "{p} joined {c} as a {j}."), ("works_at", "{p} is employed at {c}."),
    ("lives_in", "{p} has an apartment in {city}."), ("lives_in", "{p} resides in {city}."),
    ("married_to", "{p} and {q} got married in {y}."), ("married_to", "{q} is {p}'s husband."),
    ("owns", "{p} bought a {car}."), ("owns", "{p} has a cat named {pet}."),
    ("located_in", "{c} has its main office in {city}."), ("located_in", "{org} sits in downtown {city}."),
    ("scheduled_for", "The offsite is planned for {d}."), ("scheduled_for", "{proj} goes live on {d}."),
    ("uses", "{p} writes code in {lang}."), ("uses", "{sys} relies on {db}."),
    ("parent_of", "{p} is {q}'s mother."), ("parent_of", "{q} is the son of {p}."),
]


def ontology_items(rng, n):
    """-> list of (state, question, options, allowed, group)."""
    out = []
    for g in range(n):
        grp = f"ont{g}"
        if rng.random() < 0.45:
            kind = rng.choice(list(TYPES))
            mention = {
                "person": lambda: f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                "organization": lambda: rng.choice(COMPANIES + ORGS),
                "location": lambda: rng.choice(CITIES),
                "event": lambda: rng.choice(["the quarterly offsite", "Leila's wedding", "the product launch party",
                                             "the marathon", "the hackathon", "the parent-teacher evening"]),
                "product": lambda: rng.choice(EDITORS + LANGS[:3] + ["Slack", "Jira", "Kubernetes", "an iPad", "the Kindle"]),
                "date": lambda: date(rng) + rng.choice(["", ", 2025", " next year"]),
            }[kind]()
            ctx = rng.choice(["The user mentioned {m} in the chat.", "Note: {m} came up during the call.",
                              "{m} was discussed at length."]).format(m=mention)
            ctx = fix_case(ctx)
            keys = rng.sample(list(TYPES), 4)
            if kind not in keys:
                keys[rng.randrange(4)] = kind
            opts = [TYPES[k] for k in keys]
            out.append((ctx, f"What type of entity is \"{mention}\"?", opts, [keys.index(kind)], grp))
        else:
            rel, tpl = rng.choice(REL_FACTS)
            fact = tpl.format(p=rng.choice(FIRST), q=rng.choice(FIRST), c=rng.choice(COMPANIES), j=rng.choice(JOBS),
                              city=rng.choice(CITIES), y=year(rng, 1990, 2023), car=rng.choice(CARS),
                              pet=rng.choice(PETS)[1], org=rng.choice(ORGS), d=date(rng), proj=rng.choice(PROJECTS),
                              lang=rng.choice(LANGS), sys=fix_case(sys_subject(rng)), db=rng.choice(DBS))
            keys = rng.sample(list(RELS), 4)
            if rel not in keys:
                keys[rng.randrange(4)] = rel
            out.append((fix_case(fact), "Which relation does this fact express?", [RELS[k] for k in keys],
                        [keys.index(rel)], grp))
    return out
