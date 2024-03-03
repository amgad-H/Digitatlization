from typing import List


class Erziehungsberechtigter:
    def __init__(self, dict):
        self.__dict__ = dict

    def __init__(self, type, anrede, title, akad_grad, vorname, zweiter_vorname, nachname, akad_grad_nach, land, plz, gemeinde, strasse, hausnummer, telefonnummer1, telefonnummer2, email, schueler_wohnt_hier, ist_erziehungsberechtigt):
        self.type = type
        self.anrede = anrede
        self.title = title
        self.akad_grad = akad_grad
        self.vorname = vorname
        self.zweiter_vorname = zweiter_vorname
        self.nachname = nachname
        self.akad_grad_nach = akad_grad_nach
        self.land = land
        self.plz = plz
        self.gemeinde = gemeinde
        self.strasse = strasse
        self.hausnummer = hausnummer
        self.telefonnummer1 = telefonnummer1
        self.telefonnummer2 = telefonnummer2
        self.email = email
        self.schueler_wohnt_hier = schueler_wohnt_hier
        self.ist_erziehungsberechtigt = ist_erziehungsberechtigt

class Bewerber:
    def __init__(self, dict):
        self.__dict__ = dict

    def __init__(self, nachname, vornamen, geschlecht, geburtsdatum, geburtsstaat, staatsbuergerschaft, religion, muttersprache, alltagssprache, svn_kurz, telefonnummer, wunsch_abteilung, alternativ_abteilung, alternativ_abteilung2, vorschule_jahre, volkschule_jahre, mittelschule_jahre, ahs_jahre, poly_jahre, sonstige_jahre, herkunftsschule_name, herkunftsschule_typ, geschwister_an_der_schule: bool, verhalten7sst, original_jahreszeugnis: bool, erziehungsberechtigte: List[Erziehungsberechtigter]):
        self.nachname = nachname
        self.vornamen = vornamen
        self.wunsch_abteilung = wunsch_abteilung
        self.alternativ_abteilung = alternativ_abteilung
        self.alternativ_abteilung2 = alternativ_abteilung2
        self.geschlecht = geschlecht
        self.svn_kurz = svn_kurz
        self.geburtsdatum = geburtsdatum
        self.telefonnummer = telefonnummer
        self.vorschule_jahre = vorschule_jahre
        self.volkschule_jahre = volkschule_jahre
        self.mittelschule_jahre = mittelschule_jahre
        self.ahs_jahre = ahs_jahre
        self.poly_jahre = poly_jahre
        self.sonstige_jahre = sonstige_jahre
        self.herkunftsschule_typ = herkunftsschule_typ
        self.herkunftsschule_name = herkunftsschule_name
        self.geburtsstaat = geburtsstaat
        self.staatsbuergerschaft = staatsbuergerschaft
        self.muttersprache = muttersprache
        self.alltagssprache = alltagssprache
        self.religion = religion
        self.geschwister_an_der_schule = geschwister_an_der_schule
        self.verhalten7sst = verhalten7sst
        self.original_jahreszeugnis = original_jahreszeugnis
        self.erziehungsberechtigte = erziehungsberechtigte

    def unpack_guardians(self):
        if len(self.erziehungsberechtigte) > 0 and isinstance(self.erziehungsberechtigte[0], dict):
            self.erziehungsberechtigte = [Erziehungsberechtigter(**e) for e in self.erziehungsberechtigte]

    def pack_guardians(self):
        if len(self.erziehungsberechtigte) > 0 and isinstance(self.erziehungsberechtigte[0], Erziehungsberechtigter):
            self.erziehungsberechtigte = [e.__dict__ for e in self.erziehungsberechtigte]

    def __str__(self):
        return f"{self.first_name} {self.family_name}"

    @staticmethod
    def getBewerber(linked_data):
        nachname = linked_data.get('nachname', '')
        vornamen = linked_data.get('vornamen', '')
        geschlecht = linked_data.get('geschlecht', '')
        geburtsdatum = linked_data.get('geburtsdatum', '')
        geburtsstaat = linked_data.get('geburtsstaat', '')
        staatsbuergerschaft = linked_data.get('staatsbuergerschaft', '')
        religion = linked_data.get('religion', '')
        muttersprache = linked_data.get('muttersprache', '')
        alltagssprache = linked_data.get('alltagssprache', '')
        svn_kurz = linked_data.get('svn_kurz', '')
        telefonnummer = linked_data.get('telefonnummer', '')
        wunsch_abteilung = linked_data.get('wunsch_abteilung', '')
        alternativ_abteilung = linked_data.get('alternativ_abteilung', '')
        alternativ_abteilung2 = linked_data.get('alternativ_abteilung2', '')
        vorschule_jahre = linked_data.get('vorschule_jahre', '')
        volkschule_jahre = linked_data.get('volkschule_jahre', '')
        mittelschule_jahre = linked_data.get('mittelschule_jahre', '')
        ahs_jahre = linked_data.get('ahs_jahre', '')
        poly_jahre = linked_data.get('poly_jahre', '')
        sonstige_jahre = linked_data.get('sonstige_jahre', '')
        herkunftsschule_name = linked_data.get('herkunftsschule_name', '')
        herkunftsschule_typ = linked_data.get('herkunftsschule_typ', '')
        geschwister_an_der_schule = linked_data.get('geschwister_an_der_schule', False)
        verhalten7sst = linked_data.get('verhalten7sst', '')
        original_jahreszeugnis = linked_data.get('original_jahreszeugnis', False)

        # Extract Erziehungsberechtigter data
        erziehungsberechtigte_data = linked_data.get('erziehungsberechtigte', [])
        erziehungsberechtigte_objects = [Erziehungsberechtigter(**data) for data in erziehungsberechtigte_data]

        # Create a Bewerber object with the extracted data
        bewerber = Bewerber(
            nachname=nachname,
            vornamen=vornamen,
            geschlecht=geschlecht,
            geburtsdatum=geburtsdatum,
            geburtsstaat=geburtsstaat,
            staatsbuergerschaft=staatsbuergerschaft,
            religion=religion,
            muttersprache=muttersprache,
            alltagssprache=alltagssprache,
            svn_kurz=svn_kurz,
            telefonnummer=telefonnummer,
            wunsch_abteilung=wunsch_abteilung,
            alternativ_abteilung=alternativ_abteilung,
            alternativ_abteilung2=alternativ_abteilung2,
            vorschule_jahre=vorschule_jahre,
            volkschule_jahre=volkschule_jahre,
            mittelschule_jahre=mittelschule_jahre,
            ahs_jahre=ahs_jahre,
            poly_jahre=poly_jahre,
            sonstige_jahre=sonstige_jahre,
            herkunftsschule_name=herkunftsschule_name,
            herkunftsschule_typ=herkunftsschule_typ,
            geschwister_an_der_schule=geschwister_an_der_schule,
            verhalten7sst=verhalten7sst,
            original_jahreszeugnis=original_jahreszeugnis,
            erziehungsberechtigte=erziehungsberechtigte_objects
        )
        bewerber = Bewerber('', '', 'm', '12.03.1765', 'Staat', 'Bürger', 'Kein', 'Deutsch', 'Englisch', '6875', '06678420763', '', '', '', 4, 3, 5, 2, 4, 3, 'Random', 'MS', False, '', False, [])
        return bewerber