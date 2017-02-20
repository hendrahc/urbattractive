class UsersController < ApplicationController

  def new
     @nationalities = ["Dutch","Indonesian","Others"]
     @title = "Register"
     @user = User.new
  end

  def show
  end

  def create  
    @user = User.new(user_params)
    if @user.save
      redirect_to intro_part1_path
      #redirect_to ready_path
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
    params.require(:user).permit(:content1, :content2, :id, :campaign_id, :start_time, 
      :name, :email, :gender, :age)
  end

end
