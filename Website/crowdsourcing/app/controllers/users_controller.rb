class UsersController < ApplicationController

  def new
     if session[:userid]
          redirect_to intro_part1_path
     end

     session[:part] = 1
     @nationalities = ["- Please select -","Afghan","Albanian","Algerian","American","Andorran","Angolan","Antiguans","Argentinean","Armenian","Australian","Austrian","Azerbaijani","Bahamian","Bahraini","Bangladeshi","Barbadian","Barbudans","Batswana","Belarusian","Belgian","Belizean","Beninese","Bhutanese","Bolivian","Bosnian","Brazilian","British","Bruneian","Bulgarian","Burkinabe","Burmese","Burundian","Cambodian","Cameroonian","Canadian","Cape Verdean","Central African","Chadian","Chilean","Chinese","Colombian","Comoran","Congolese","Costa Rican","Croatian","Cuban","Cypriot","Czech","Danish","Djibouti","Dominican","Dutch","East Timorese","Ecuadorean","Egyptian","Emirian","Equatorial Guinean","Eritrean","Estonian","Ethiopian","Fijian","Filipino","Finnish","French","Gabonese","Gambian","Georgian","German","Ghanaian","Greek","Grenadian","Guatemalan","Guinea-Bissauan","Guinean","Guyanese","Haitian","Herzegovinian","Honduran","Hungarian","I-Kiribati","Icelander","Indian","Indonesian","Iranian","Iraqi","Irish","Irish","Israeli","Italian","Ivorian","Jamaican","Japanese","Jordanian","Kazakhstani","Kenyan","Kittian and Nevisian","Kuwaiti","Kyrgyz","Laotian","Latvian","Lebanese","Liberian","Libyan","Liechtensteiner","Lithuanian","Luxembourger","Macedonian","Malagasy","Malawian","Malaysian","Maldivan","Malian","Maltese","Marshallese","Mauritanian","Mauritian","Mexican","Micronesian","Moldovan","Monacan","Mongolian","Moroccan","Mosotho","Motswana","Mozambican","Namibian","Nauruan","Nepalese","Netherlander","New Zealander","Ni-Vanuatu","Nicaraguan","Nigerian","North Korean","Northern Irish","Norwegian","Omani","Pakistani","Palauan","Panamanian","Papua New Guinean","Paraguayan","Peruvian","Polish","Portuguese","Qatari","Romanian","Russian","Rwandan","Saint Lucian","Salvadoran","Samoan","San Marinese","Sao Tomean","Saudi","Scottish","Senegalese","Serbian","Seychellois","Sierra Leonean","Singaporean","Slovakian","Slovenian","Solomon Islander","Somali","South African","South Korean","Spanish","Sri Lankan","Sudanese","Surinamer","Swazi","Swedish","Swiss","Syrian","Taiwanese","Tajik","Tanzanian","Thai","Togolese","Tongan","Trinidadian","Tunisian","Turkish","Tuvaluan","Ugandan","Ukrainian","Uruguayan","Uzbekistani","Venezuelan","Vietnamese","Welsh","Yemenite","Zambian","Zimbabwean","Other"]
     @title = "Register"
     @user = User.new
  end

  def show
  end

  def create
    @user = User.new(user_params)
    if @user.save
      session[:userid] = @user.id
      session[:user] = @user.name
	  session[:campaign] = @user.campaign_id
      session[:locs] = CampaignSet.find(session[:campaign]).locs

      #randomize images
       session[:locs] = session[:locs].split(" ")
       session[:images] = Image.where(Image.arel_table[:loc_id].in(session[:locs])).ids.shuffle()
	   (session[:locs]).push("9999")
       session[:img_num] = 0;
       session[:part] = 1

       #randomize golden questions
       session[:goldens] = GoldenQuestion.ids.shuffle()
       session[:golden_num] = 0
       session[:checkimage] = 0

      redirect_to intro_part1_path
      #redirect_to root_path
    else
      render 'new'
    end
  end

  def content
    @title = "Question"
    if session[:content].nil?
      session[:content] = 1
    else 
      session[:content] = 2
    end
  end

  def update
    if session[:content] == 1
      @user = User.find_by!(name: session[:userid])
      @user.content1 = params[:answer]
      @user.update_attribute(:content1, @user.content1)
      session[:content] = 2
      session[:img_num] = session[:img_num].to_i + 1
      redirect_to show_part1_path
    else 
      @user = User.find_by!(name: session[:userid])
      @user.content2 = params[:answer]
      @user.update_attribute(:content2, @user.content2)
      session[:content] = nil
      session[:img_num] = session[:img_num].to_i + 1
      redirect_to show_part1_path
    end 
  end

  def crowdsourceuser
    @user = User.new(:name => session[:userid], :email => "", :campaign_id => session[:campaign],)
    @user.age = params[:user][:age]
    @user.gender = params[:user][:gender]
    @user.content1=""
    @user.content2=""
    @user.start_time = Time.now.strftime("%I:%M:%S %z")
    @user.save
    redirect_to ready_path
#    redirect_to intro_path
  end

private

  def user_params
    params.require(:user).permit(:campaign_id, :name, :email, :gender, :age, :nationality, :start_time)
  end

end
