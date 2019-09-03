
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo

class KeywordForm(FlaskForm):
	first_keyword = StringField('first_keyword', validators=[DataRequired()])
	second_keyword = StringField('second_keyword', validators=[DataRequired()])
	submit = SubmitField('Analyse')