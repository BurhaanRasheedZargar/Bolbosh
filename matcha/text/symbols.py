""" from https://github.com/keithito/tacotron

Defines the set of symbols used in text input to the model.
"""
_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)

ALL_CHARACTERS = ['', ' ', ' - ', '"', "'", '-', '...', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '،', '؛', '؟', 'ؠ', 'ء', 'آ', 'أ', 'ؤ', 'ئ', 'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ل', 'لا', 'م', 'ن', 'و', 'َ', 'ُ', 'ِ', 'ٔ', 'ٕ', 'ٖ', 'ٗ', 'ٚ', 'ٛ', 'ٟ', '٪', '٫', 'ٰ', 'ٹ', 'پ', 'چ', 'ڈ', 'ڑ', 'ژ', 'ک', 'گ', 'ں', 'ھ', 'ہ', 'ۂ', 'ۃ', 'ۄ', 'ی', 'ۍ', 'ے', 'ۓ', '۔', '۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹']
# Export all symbols:
# symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + list(ALL_CHARACTERS) 
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + ALL_CHARACTERS




# Special symbol ids
SPACE_ID = symbols.index(" ")